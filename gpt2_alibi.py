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
from muon import Muon


@dataclass
class GPTConfig:
    # max sequence length
    vocab_size: int = 16389

    # MUSIC
    # 9/10/640 = 47.38M, fails (needs more attention heads? second guess depth too low, then unlikely but width too high / too high for depth)
    # 11/12/576 = 45.86M, works.
    # 8/16/640 = 43.28M, works, initially faster, more stable, but "final" loss ~3.9% lower.
    # 8/12/576 = 36M, works, at least short training (all I've tested).
    # *11/14/672 = 60.72M
    # *Try 14/16/704 (possibly 13 layers depending max ctx)
    n_layer: int = 8 #11
    n_head: int = 12 #14
    n_embd: int = 576 #672
    dropout: float = 0.04
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    gradient_checkpointing: bool = False
    # How many segments to split the (main Block) layers into for gradient checkpointing.
    # More segments = more VRAM savings and slower.
    gradient_checkpointing_num_segments: int = 2

    # tanh clamp logits to this value on the forward pass
    # Set to None to disable
    logit_clamp: float = 30.0

    use_value_residual: bool = True
    use_learnable_lambda: bool = False
    use_embed_shortcut: bool = True

    def __post_init__(self):
        if self.use_learnable_lambda and not self.use_value_residual:
            raise ValueError("Learnable Lambda requires Value Residual to be enabled")


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
        self.use_value_residual = config.use_value_residual
        self.use_learnable_lambda = config.use_learnable_lambda
        if self.use_learnable_lambda:
            self.lambda_param = nn.Parameter(torch.tensor(0.5))

        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.zero_init = True  # See GPT._init_weights
        # ALiBi slopes
        self.register_buffer("alibi_slope", get_alibi_slope(config.n_head))

        # Regularization layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, v1=None):
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension

        # Project to queries, keys, values
        qkv = self.c_attn(x.to(torch.bfloat16))  # Shape: (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=2)  # Each is (B, T, C)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).to(torch.bfloat16)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).to(torch.bfloat16)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).to(torch.bfloat16)  # (B, n_head, T, head_dim)

        if self.use_value_residual:
            if v1 is None:
                v1 = v
            if self.use_learnable_lambda:
                v = self.lambda_param * v + (1 - self.lambda_param) * v1.view_as(v)
            else:
                v = 0.5 * v + 0.5 * v1.view_as(v)

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

        self.use_embed_shortcut = config.use_embed_shortcut
        if self.use_embed_shortcut:
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

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

    def forward(self, x, v1=None, x0=None):
        if self.use_embed_shortcut and x0 is not None:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(self.ln_1(x), v1)
        x = x + self.mlp(self.ln_2(x).to(torch.bfloat16))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.use_value_residual = config.use_value_residual
        self.use_embed_shortcut = config.use_embed_shortcut

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

        # Optimizers and schedulers will be set in configure_optimizers
        self.optimizers = None
        self.schedulers = None
        self.get_clip_func = None

        # Report number of parameters
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Check if this linear layer should be zero-initialized
            if hasattr(module, 'zero_init') and module.zero_init:
                torch.nn.init.zeros_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            else:
                # Normal initialization for other linear layers
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
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
        B, T = idx.size()

        x = self.transformer.wte(idx).to(torch.bfloat16)  # (B, T, C)
        x0 = x if self.use_embed_shortcut else None
        v1 = None

        if self.config.gradient_checkpointing:
            # Modify checkpoint_sequential to pass v1 and x0
            def custom_forward(*inputs):
                return checkpoint_sequential(self.transformer.h, self.config.gradient_checkpointing_num_segments,
                                             *inputs, use_reentrant=False)

            x = custom_forward(x, v1, x0)
        else:
            for i, block in enumerate(self.transformer.h):
                if i == 0 and self.use_value_residual:
                    _, _, v1 = block.attn.c_attn(x).chunk(3, dim=2)
                    v1 = v1.view(B, T, block.attn.n_head, block.attn.n_embd // block.attn.n_head).transpose(1, 2)
                x = block(x, v1, x0)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        del x
        if self.config.logit_clamp:
            logits = self.config.logit_clamp * torch.tanh(logits / self.config.logit_clamp)

        loss = None
        if targets is not None:
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        # Disabled because of inconvenience of the shape size vs minimal speed improvement
        #else:
        #    Inference optimization: only compute logits for the last token
        #    logits = self.lm_head(x[:, -1, :])  # (B, vocab_size)

        return logits, loss

    def configure_optimizers(self, weight_decay, weight_decay_attn, learning_rate_adamw, learning_rate_muon, learning_rate_residual, muon_momentum_warmup, device_type, get_lr_func, get_clip_func, ddp=False, ddp_rank=0, ddp_world_size=1,
                             betas=(0.9, 0.95)):
        # Separate parameters by optimizer
        adamw_params = []
        muon_params = []
        residual_params = []

        # Collect parameters from lm_head and wte (they share weights) for AdamW
        adamw_params.extend(
            [(name, p) for name, p in self.lm_head.named_parameters() if p.requires_grad]
        )
        adamw_params.extend(
            [(name, p) for name, p in self.transformer.ln_f.named_parameters() if p.requires_grad]
        )

        # Collect parameters from transformer blocks
        for block in self.transformer.h:
            for name, param in block.named_parameters():
                if not param.requires_grad:
                    continue
                if (self.config.use_learnable_lambda and 'lambda_param' in name) or \
                        (self.config.use_embed_shortcut and 'lambdas' in name):
                    residual_params.append(param)
                elif param.dim() == 2:
                    # Parameters with dim==2 are assigned to Muon optimizer
                    muon_params.append(param)
                else:
                    # Other parameters go to AdamW
                    adamw_params.append((name, param))

        # Separate AdamW parameters for weight decay
        decay_attn_params = []
        decay_other_params = []
        nodecay_params = []

        for name, param in adamw_params:
            if param.dim() >= 1 and not any(nd in name.lower() for nd in ['bias', 'ln_', 'layernorm']):
                if 'attn' in name.lower():
                    # Decay attention parameters with weight_decay_attn
                    decay_attn_params.append(param)
                else:
                    # Decay other parameters with weight_decay
                    decay_other_params.append(param)
            else:
                # Do not decay biases and LayerNorm weights
                nodecay_params.append(param)

        # Create optimizer parameter groups
        adamw_groups = []
        if decay_attn_params:
            adamw_groups.append({
                'params': decay_attn_params,
                'weight_decay': weight_decay_attn
            })
        if decay_other_params:
            adamw_groups.append({
                'params': decay_other_params,
                'weight_decay': weight_decay
            })
        if nodecay_params:
            adamw_groups.append({
                'params': nodecay_params,
                'weight_decay': 0.0
            })
        if residual_params:
            adamw_groups.append({
                'params': residual_params,
                'weight_decay': 0.0,
                'lr': learning_rate_residual   # Separate learning rate for residual parameters
            })

        # Print information about the number of parameters
        total_params = self.get_num_params()
        num_decay_attn_params = sum(p.numel() for p in decay_attn_params)
        num_decay_other_params = sum(p.numel() for p in decay_other_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_residual_params = sum(p.numel() for p in residual_params)
        num_adamw_params = num_decay_attn_params + num_decay_other_params + num_nodecay_params + num_residual_params
        num_muon_params = sum(p.numel() for p in muon_params)

        print(f"Total model parameters: {total_params}")
        print(f"AdamW parameters: {num_adamw_params} ({num_adamw_params / total_params * 100:.2f}% of total)")
        print(f"  Decayed attention parameters: {num_decay_attn_params} ({num_decay_attn_params / total_params * 100:.2f}% of total)")
        print(f"  Decayed other parameters: {num_decay_other_params} ({num_decay_other_params / total_params * 100:.2f}% of total)")
        print(f"  Non-decayed parameters: {num_nodecay_params} ({num_nodecay_params / total_params * 100:.2f}% of total)")
        print(f"  Residual parameters: {num_residual_params} ({num_residual_params / total_params * 100:.2f}% of total)")
        print(f"Muon parameters: {num_muon_params} ({num_muon_params / total_params * 100:.2f}% of total)")

        # Create optimizers
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        optimizer1 = torch.optim.AdamW(adamw_groups, lr=learning_rate_adamw, betas=betas, fused=use_fused)
        optimizer2 = Muon(muon_params, lr=learning_rate_muon, momentum=0.95, momentum_warmup=muon_momentum_warmup, ddp=ddp, rank=ddp_rank,
                          world_size=ddp_world_size)

        self.optimizers = [optimizer1, optimizer2]
        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda it, idx=i: get_lr_func(it)[idx])
            for i, opt in enumerate(self.optimizers)
        ]
        self.get_clip_func = get_clip_func

    def move_optimizers_to_cpu(self):
        """Move optimizer states to CPU"""
        for opt in self.optimizers:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
            # For Muon optimizer specifically (if it has device-specific attributes)
            if hasattr(opt, 'to_cpu'):
                opt.to_cpu()

    def move_optimizers_to_device(self, device):
        """Move optimizer states to specified device"""
        for opt in self.optimizers:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            # For Muon optimizer specifically (if it has device-specific attributes)
            if hasattr(opt, 'to_device'):
                opt.to_device(device)

    def step(self, clip_adamw=True, clip_muon=True, skipped_last=False):
        clip_val_adamw, clip_val_muon = self.get_clip_func()
        norm_adamw = torch.nn.utils.clip_grad_norm_(
            # Turns out clipping the non-decayed parameters is a bad idea
            self.optimizers[0].param_groups[0]['params'],  # + self.optimizers[0].param_groups[1]['params'],
            clip_val_adamw if clip_adamw else float('inf')
        )
        norm_muon = torch.nn.utils.clip_grad_norm_(self.optimizers[1].param_groups[0]['params'], clip_val_muon if clip_muon else float('inf'))

        skip_step = norm_adamw > 20 * clip_val_adamw or norm_muon > 40 * clip_val_muon
        skip_step = skip_step and not skipped_last
        if not skip_step:
            for opt, sched in zip(self.optimizers, self.schedulers):
                opt.step()
                sched.step()

        self.zero_grad(set_to_none=True)
        return norm_adamw.item(), norm_muon.item(), skip_step