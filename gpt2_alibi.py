from collections import namedtuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint_sequential
import inspect
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.layer_norm import LigerLayerNorm
import math


@dataclass
class GPTConfig:
    # max sequence length
    vocab_size: int = 16389

    # MUSIC
    # 9/10/640 = 47.38M, fails (needs more attention heads? second guess depth too low, then unlikely but width too high / too high for depth)
    # 11/12/576 = 45.86M, works.
    # 8/16/640 = 43.28M, works, initially faster, more stable, but "final" loss ~3.9% lower.
    # 11/14/672 = 60.72M
    # Try 14/16/704 (possibly 13 layers depending max ctx)
    n_layer: int = 11
    n_head: int = 14
    n_embd: int = 672
    dropout: float = 0.04
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    gradient_checkpointing: bool = False
    # How many segments to split the (main Block) layers into for gradient checkpointing.
    # More segments = more VRAM savings and slower.
    gradient_checkpointing_num_segments: int = 2


def get_alibi_slope(num_heads):
    # ALiBi slope calculation based on the number of heads
    # Reference: https://arxiv.org/abs/2108.12409
    x = (2 ** 8) ** (1 / num_heads)
    return torch.tensor([1 / (x ** (i + 1)) for i in range(num_heads)], dtype=torch.bfloat16).view(num_heads, 1, 1)


class ALiBiAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # ALiBi slopes
        self.register_buffer("alibi_slope", get_alibi_slope(config.n_head))

        # Regularization layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension

        # Project to queries, keys, values
        qkv = self.c_attn(x.to(torch.bfloat16))  # Shape: (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=2)  # Each is (B, T, C)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).to(torch.bfloat16)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).to(torch.bfloat16)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).to(torch.bfloat16)  # (B, n_head, T, head_dim)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(C // self.n_head)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)
        att.to(torch.bfloat16)

        # Compute absolute relative positions
        i = torch.arange(T, device=x.device, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)
        j = torch.arange(T, device=x.device, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(-1)
        # shape: (1, n_head, T, T)
        alibi_bias = (i - j).clamp(min=0) * self.alibi_slope.view(self.n_head, 1, 1)
        del i, j

        # Add ALiBi bias
        att.add_(alibi_bias)

        # Causal mask: mask future tokens
        mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bfloat16)).bool()
        att = att.masked_fill(~mask, float("-inf"))

        # Softmax and dropout
        att = F.softmax(att, dim=-1, dtype=torch.bfloat16)  # If experiencing instabilities, switching this to float32 is a good first guess.
        att = self.attn_dropout(att).to(torch.bfloat16)

        # Attention output
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C).to(torch.bfloat16)  # (B, T, C)

        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))  # (B, T, C)
        return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LigerLayerNorm(config.n_embd, eps=1e-5)
        self.ln_2 = LigerLayerNorm(config.n_embd, eps=1e-5)
        self.attn = ALiBiAttention(config)
        GeluConfig = namedtuple("GeluConfig", ["hidden_size", "intermediate_size", "hidden_act"])
        self.mlp = LigerGEGLUMLP(GeluConfig(
            hidden_size=config.n_embd,
            intermediate_size=2 * config.n_embd,
            hidden_act="gelu_pytorch_tanh"
        ))
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x).to(torch.bfloat16))  # If experiencing instabilities, switching this to float32 is a good second guess (cast this final result to float32 and that failing remove the bfloat16 cast).
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # ALiBi does not use a wpe
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LigerLayerNorm(config.n_embd, eps=1e-5),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie the lm_head weights to the wte weights
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        self.criterion = LigerCrossEntropyLoss()

        # Report number of parameters
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # Have tested without this init, but not with yet!
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LigerGEGLUMLP):
            torch.nn.init.normal_(module.gate_proj.weight, mean=0.0, std=0.02 / (2 * self.config.n_layer) ** 0.25)
            torch.nn.init.normal_(module.up_proj.weight, mean=0.0, std=0.02 / (2 * self.config.n_layer) ** 0.25)
            torch.nn.init.normal_(module.down_proj.weight, mean=0.0, std=0.02)
        elif isinstance(module, LigerLayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        #B, T = idx.size()

        x = self.transformer.wte(idx).to(torch.bfloat16)  # (B, T, C)

        if self.config.gradient_checkpointing:
            x = checkpoint_sequential(self.transformer.h, self.config.gradient_checkpointing_num_segments, x, use_reentrant=False)
        else:
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        del x

        loss = None
        if targets is not None:
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        # Disabled because of inconvenience of the shape size vs minimal speed improvement
        #else:
        #    Inference optimization: only compute logits for the last token
        #    logits = self.lm_head(x[:, -1, :])  # (B, vocab_size)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, betas=(0.9, 0.95)):
        # Start with all trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate parameters for weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Check for fused optimizer support
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
