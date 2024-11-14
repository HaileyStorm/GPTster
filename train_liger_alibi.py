import os
import math
import time
import random
import torch
from grokfast import gradfilter_ema
from gpt2_alibi import GPTConfig, GPT
from dataloader import DataLoaderLite
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.nn import functional as F
from offset_tokenizer import AudioTokenizer
from scipy.io.wavfile import write
from traceback import format_exc
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from enum import Enum, auto
import signal

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


class LrFunc(Enum):
    WSD = auto()
    WSD_Inv = auto()
    Cosine = auto()


# ------------------------------
# HYPER-PARAMETERS
# ------------------------------

wandb_project = "MusicGPT-Small"

# This gets overwritten (loaded from the checkpoint) if resume == True
config = GPTConfig(
    n_layer=14,
    n_head=16,
    n_embd=704,
    dropout=0.04,
    gradient_checkpointing=False,
    gradient_checkpointing_num_segments=2,
    logit_clamp=30.0,  # None to disable
    use_value_residual=True,
    use_learnable_lambda=True,
    use_embed_shortcut=True
)

# sequence length (note that the model does not have a block size / max sequence length - a change from the original version of this code, so there's some left-overs from that still)
T = 6483
total_batch_size = T*20
# micro batch size (T*B, and ofc model dims, determines VRAM use; total_batch_size//B is grad accum steps)
B = 1

inference_batch_size = 1
assert inference_batch_size <= B

weight_decay = 0.05
# Currently not used (Muon covers all ALiBi parameters and doesn't use decay)
weight_decay_attn = 0.05

max_lr_adamw = 4.25e-4
max_lr_muon = max_lr_adamw * 0.535
lr_residual = 0.0025
init_lr_pct = 0.1
min_lr_pct = 0.015
warmup_steps_wsd = 0
warmup_steps_cosine = 333
# Warmup Muon momentum value over this many steps, from 0.895x to 1x (by default, 0.85->0.95). Set to 0 to disable.
muon_momentum_warmup = 250
warmdown_pct_wsd = 2.0 / 7.0
lr_func_adamw = LrFunc.Cosine
lr_func_muon = LrFunc.WSD_Inv

# original (music_data_shuffled) base 2.5
# pop2 4.77
# pop3 7
num_epochs = 21

save_every = 5000 #2500
log_dir = "log_alibi"
checkpoint_prefix = "muon"  # "model"
resume = False
resume_from = './log_alibi/model_s7619.pt'
# Whether to reset (or load from checkpoint) the optimizer. Also resets norms&loss windows.
reset_optimizer = False
# Whether to reset (or load from checkpoint) the schedule (the step number & therefore learning rate, best val loss, etc.)
reset_schedule = False

use_compile = True

clip_adamw = False
clip_muon = False
grad_clip_percentile = 0.125
grad_clip_initial_adamw = 0.5
grad_clip_min_adamw = 1e-3
grad_clip_max_adamw = 0.85
grad_clip_initial_muon = 0.85
grad_clip_min_muon = 1e-3
grad_clip_max_muon = 2.25
norms_window_size = 250
# Decrease lr when norm percentile & loss are increasing
max_clip_slope = 1.1
lr_adj_rate = 0.925  # effectively, max_lr = max_lr * lr_adj_rate every norms_window_size/3 steps while conditions met

grok_enabled = False
grok_start_divergence = 1.035
divergence_window_size = 5
grok_warmup_steps = 2000
grok_alpha = 0.925
grok_lamb = 1.1

# ------------------------------
# END HYPER-PARAMETERS
# ------------------------------


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# pytorch can be serious about its device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

if master_process:
    wandb.init(project=wandb_project)

chunk_size = T // 3
print(f"Chunk size: {chunk_size}")

assert total_batch_size % (B * T * ddp_world_size) == 0, "Make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", ddp=ddp, master_process=master_process, critical_divisor=chunk_size)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", ddp=ddp, master_process=master_process, critical_divisor=chunk_size)

# Was 15-64k. In theory our goal is 610k +  (~2TB of 32khz audio if it were single epoch, or ~57.5GB tokenized [total, obviously some # epochs > 1 is fine, at least 6 & probably more w/ more data])
max_steps = int((num_epochs * train_loader.total_tokens) // total_batch_size)
print(f"Max steps: {max_steps}, Warmup steps: {max(warmup_steps_wsd, warmup_steps_cosine, muon_momentum_warmup)}")

torch.set_float32_matmul_precision('high')

pause_training = False


def pause_handler(signum, frame):
    global pause_training
    if not pause_training:
        if master_process:
            print("\nPausing after current step completes...")
        pause_training = True


def resume_handler(signum, frame):
    global pause_training
    if pause_training:
        if master_process:
            print("\nResuming training...")
        pause_training = False


def get_lr_wsd(it):
    """
    WSD learning rate schedule.
    - Warmup: Linear increase from init_lr_pct to 1.0 over warmup_steps_muon.
    - Stable Phase: Stable at 1.0.
    - Warmdown/Decay Phase: Linear decay from 1.0 to min_lr_pct.

    Args:
        it (int): Current iteration step.

    Returns:
        float: Learning rate for the current step.
    """
    # Warmup phase
    if it < warmup_steps_wsd:
        return init_lr_pct + (1.0 - init_lr_pct) * (it / warmup_steps_wsd)

    # Calculate start of warmdown
    warmdown_start = max_steps * (1 - warmdown_pct_wsd)

    # Stable phase
    if it < warmdown_start:
        return 1.0

    # Warmdown/decay phase
    warmdown_steps = max_steps * warmdown_pct_wsd
    decay_ratio = (it - warmdown_start) / warmdown_steps
    return (1 - decay_ratio) * (1 - min_lr_pct) + min_lr_pct


def get_lr_wsd_inv(it):
    """
    WSD learning rate schedule with inverse-square-root stable phase.
    - Warmup: Linear increase from init_lr_pct to 1.0 over warmup_steps_muon.
    - Stable Phase: Inverse square root decay from 1.0 to 0.85.
    - Warmdown/Decay Phase: Linear decay from 0.85 to min_lr_pct.

    Args:
        it (int): Current iteration step.

    Returns:
        float: Learning rate for the current step.
    """
    # Warmup phase
    if it < warmup_steps_wsd:
        return init_lr_pct + (1.0 - init_lr_pct) * (it / warmup_steps_wsd)

    # Calculate start of warmdown
    warmdown_start = max_steps * (1 - warmdown_pct_wsd)

    # Stable phase with inverse square root decay from 1.0 to 0.85
    if it < warmdown_start:
        # Progress between warmup and warmdown
        progress = (it - warmup_steps_wsd) / (warmdown_start - warmup_steps_wsd)
        # Calculate the decay factor using inverse square root
        # Ensures lr=1.0 at progress=0 and approaches 0.85 as progress approaches 1
        b = (1 / 0.85) ** 2 - 1  # b ≈ 0.3843
        lr = 1.0 / math.sqrt(1 + b * progress)
        return lr

    # Warmdown/Decay phase: Linear decay from 0.85 to min_lr_pct
    warmdown_steps = max_steps * warmdown_pct_wsd
    decay_ratio = (it - warmdown_start) / warmdown_steps
    lr = (1 - decay_ratio) * 0.85 + decay_ratio * min_lr_pct
    return lr


def get_lr_cosine(it):
    """
    Learning rate scheduler for AdamW.
    - Warmup: Linear increase from init_lr_pct to 1.0 over warmup_steps_adamw.
    - Cosine Decay: Cosine decay from 1.0 to min_lr_pct over the remaining steps.

    Args:
        it (int): Current iteration step.

    Returns:
        float: Learning rate for the current step.
    """
    # Warmup phase
    if it < warmup_steps_cosine:
        return init_lr_pct + (1.0 - init_lr_pct) * (it / warmup_steps_cosine)

    # Cosine decay phase
    progress = (it - warmup_steps_cosine) / (max_steps - warmup_steps_cosine)
    if progress > 1.0:
        progress = 1.0  # Clamp progress to 1.0 to avoid negative learning rates

    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    lr = min_lr_pct + (1.0 - min_lr_pct) * cosine_decay
    return lr


def get_lrs(it):
    """
    Get the AdamW and Muon learning rate (ratio), for passing to model.step. Allows for easily switching schedules for
    each optimizer.
    :param it: The current iteration (step)
    :return: The AdamW and Muon learning rate (ratio)
    """
    adamw_func = None
    muon_func = None
    match lr_func_adamw:
        case LrFunc.Cosine:
            adamw_func = get_lr_cosine
        case LrFunc.WSD_Inv:
            adamw_func = get_lr_wsd_inv
        case LrFunc.WSD:
            adamw_func = get_lr_wsd
    match lr_func_muon:
        case LrFunc.Cosine:
            muon_func = get_lr_cosine
        case LrFunc.WSD_Inv:
            muon_func = get_lr_wsd_inv
        case LrFunc.WSD:
            muon_func = get_lr_wsd

    return float(adamw_func(it)), float(muon_func(it))


def get_clip_value():
    global max_lr_adamw, max_lr_muon, total_panic_adamw, total_panic_muon, optimizer_resets_adamw, optimizer_resets_muon, norms_window_adamw, norms_window_muon, step, grad_clip_percentile, grad_clip_min_adamw, grad_clip_max_adamw, grad_clip_min_muon, grad_clip_max_muon, grad_clip_initial_adamw, grad_clip_initial_muon

    # AdamW clip value
    if step < warmup_steps_cosine * 0.85 or len(norms_window_adamw) < norms_window_size:
        clip_val_adamw = grad_clip_initial_adamw
    else:
        clip_val_adamw = np.percentile(norms_window_adamw, grad_clip_percentile * 100)
        third = norms_window_size // 3
        p1_adamw = np.percentile(norms_window_adamw[:third], grad_clip_percentile * 100)
        p2_adamw = np.percentile(norms_window_adamw[third:2*third], grad_clip_percentile * 100)
        p3_adamw = np.percentile(norms_window_adamw[2*third:], grad_clip_percentile * 100)
        l1_adamw = np.mean(loss_window[:third])
        l2_adamw = np.mean(loss_window[third:2*third])
        l3_adamw = np.mean(loss_window[2*third:])
        if p3_adamw > p2_adamw > p1_adamw and p3_adamw / p2_adamw > max_clip_slope and p2_adamw / p1_adamw > max_clip_slope and l3_adamw > l2_adamw > l1_adamw:
            max_lr_adamw *= lr_adj_rate ** (3 / norms_window_size)
            total_panic_adamw += 1
            if total_panic_adamw % (third * 2) == 0:
                print("Too much panic: Resetting AdamW optimizer.")
                raw_model.configure_optimizers(
                    weight_decay=weight_decay,
                    weight_decay_attn=weight_decay_attn,
                    learning_rate_adamw=max_lr_adamw,
                    learning_rate_muon=max_lr_muon,
                    learning_rate_residual=lr_residual,
                    muon_momentum_warmup=muon_momentum_warmup,
                    device_type=device_type,
                    get_lr_func=get_lrs,
                    get_clip_func=get_clip_value,
                    ddp=ddp,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size
                )
                optimizer_resets_adamw += 1
            wandb.log({
                "debug/panic_adamw": 1.0,
                "debug/total_panic_adamw": total_panic_adamw,
                "debug/max_lr_adamw": max_lr_adamw,
                "debug/optimizer_resets_adamw": optimizer_resets_adamw,
            }, step=step)
        else:
            wandb.log({
                "debug/panic_adamw": 0.0,
                "debug/total_panic_adamw": total_panic_adamw,
                "debug/max_lr_adamw": max_lr_adamw,
                "debug/optimizer_resets_adamw": optimizer_resets_adamw,
            }, step=step)

        clip_val_adamw = max(grad_clip_min_adamw, min(grad_clip_max_adamw, clip_val_adamw))

    # Muon clip value
    if step < warmup_steps_wsd * 0.85 or len(norms_window_muon) < norms_window_size:
        clip_val_muon = grad_clip_initial_muon
    else:
        clip_val_muon = np.percentile(norms_window_muon, grad_clip_percentile * 100)
        third = norms_window_size // 3
        p1_muon = np.percentile(norms_window_muon[:third], grad_clip_percentile * 100)
        p2_muon = np.percentile(norms_window_muon[third:2*third], grad_clip_percentile * 100)
        p3_muon = np.percentile(norms_window_muon[2*third:], grad_clip_percentile * 100)
        l1_muon = np.mean(loss_window[:third])
        l2_muon = np.mean(loss_window[third:2*third])
        l3_muon = np.mean(loss_window[2*third:])
        if p3_muon > p2_muon > p1_muon and p3_muon / p2_muon > max_clip_slope and p2_muon / p1_muon > max_clip_slope and l3_muon > l2_muon > l1_muon:
            max_lr_muon *= lr_adj_rate ** (3 / norms_window_size)
            total_panic_muon += 1
            if total_panic_muon % (third * 2) == 0:
                print("Too much panic: Resetting Muon optimizer.")
                raw_model.configure_optimizers(
                    weight_decay=weight_decay,
                    weight_decay_attn=weight_decay_attn,
                    learning_rate_adamw=max_lr_adamw,
                    learning_rate_muon=max_lr_muon,
                    learning_rate_residual=lr_residual,
                    muon_momentum_warmup=muon_momentum_warmup,
                    device_type=device_type,
                    get_lr_func=get_lrs,
                    get_clip_func=get_clip_value,
                    ddp=ddp,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size
                )
                optimizer_resets_muon += 1
            wandb.log({
                "debug/panic_muon": 1.0,
                "debug/total_panic_muon": total_panic_muon,
                "debug/max_lr_muon": max_lr_muon,
                "debug/optimizer_resets_muon": optimizer_resets_muon,
            }, step=step)
        else:
            wandb.log({
                "debug/panic_muon": 0.0,
                "debug/total_panic_muon": total_panic_muon,
                "debug/max_lr_muon": max_lr_muon,
                "debug/optimizer_resets_muon": optimizer_resets_muon,
            }, step=step)

        clip_val_muon = max(grad_clip_min_muon, min(grad_clip_max_muon, clip_val_muon))

    return clip_val_adamw, clip_val_muon


# create model
chkpt = None
if resume:
    print(f"Resuming from {resume_from}")
    chkpt = torch.load(resume_from, map_location=torch.device('cpu'))
    config: GPTConfig = chkpt["config"]

model = GPT(config)
model.to(device)
criterion = LigerCrossEntropyLoss()
grads = None
step = 0
best_val_loss = 999.9
norms_window_adamw = []
norms_window_muon = []
loss_window = []
divergence_window = []
grok_start_step = -1
tokens_trained = 0
time_trained = 0.0
val_loss_accum = None

if resume:
    model.load_state_dict(OrderedDict([
            (key.replace('_orig_mod.', ''), value) for key, value in chkpt['model'].items()
        ]))
    if not reset_schedule:
        if "step" in chkpt:
            step = chkpt["step"]
            muon_momentum_warmup = max(0, muon_momentum_warmup - step)
            train_loader.skip_batches(step * grad_accum_steps)
            # Approximate and not too important
            val_loader.skip_batches(step * 10)
        if "val_loss" in chkpt:
            best_val_loss = chkpt["val_loss"]
        if "tokens_trained" in chkpt:
            tokens_trained = chkpt["tokens_trained"]
        if "time_trained" in chkpt:
            time_trained = chkpt["time_trained"]
    model.configure_optimizers(
        weight_decay=weight_decay,
        weight_decay_attn=weight_decay_attn,
        learning_rate_adamw=max_lr_adamw,
        learning_rate_muon=max_lr_muon,
        learning_rate_residual=lr_residual,
        muon_momentum_warmup=muon_momentum_warmup,
        device_type=device_type,
        get_lr_func=get_lrs,
        get_clip_func=get_clip_value,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size
    )
    if not reset_optimizer:
        for opt, opt_state in zip(model.optimizers, chkpt['optimizers']):
            opt.load_state_dict(opt_state)
        for sched, sched_state in zip(model.schedulers, chkpt['schedulers']):
            sched.load_state_dict(sched_state)
        if "norms_window_adamw" in chkpt:
            norms_window_adamw = chkpt["norms_window_adamw"]
        if "norms_window_muon" in chkpt:
            norms_window_muon = chkpt["norms_window_muon"]
        if "loss_window" in chkpt:
            loss_window = chkpt["loss_window"]
        if "divergence_window" in chkpt:
            divergence_window = chkpt["divergence_window"]
        if "grok_start_step" in chkpt:
            grok_start_step = chkpt["grok_start_step"]
else:
    model.configure_optimizers(
        weight_decay=weight_decay,
        weight_decay_attn=weight_decay_attn,
        learning_rate_adamw=max_lr_adamw,
        learning_rate_muon=max_lr_muon,
        learning_rate_residual=lr_residual,
        muon_momentum_warmup=muon_momentum_warmup,
        device_type=device_type,
        get_lr_func=get_lrs,
        get_clip_func=get_clip_value,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size
    )

if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


def min_p_sampling(logits, p_base):
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    # Get the probability of the top token
    p_top, _ = probs.max(dim=-1, keepdim=True)
    # Calculate the dynamic threshold
    p_threshold = p_base * p_top
    # Create a mask for tokens above the threshold
    mask = probs >= p_threshold
    # Zero out probabilities below the threshold
    filtered_probs = probs * mask
    # Renormalize the remaining probabilities
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    return filtered_probs


# Note that my default min-p temp is 0.968, but during training a bit lower helps.
def generate_tokens(model, seq_length, batch_size, prefill=None, temperature=0.955, p_base=0.0515):
    if prefill is None:
        tokens = [random.randint(0, 4095)]
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(batch_size, 1)
        x = tokens.to(device)
    else:
        x = prefill

    with torch.no_grad():
        for _ in tqdm(range(T - x.size(1)), dynamic_ncols=True, desc="Generating tokens", position=1, leave=False):
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(x)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature
            # Handle NaN and Inf values in logits
            nan_mask = torch.isnan(next_token_logits) | torch.isinf(next_token_logits)
            if nan_mask.any():
                # print("Warning: NaN or Inf values detected in logits. Replacing with very negative values.")
                next_token_logits = torch.where(nan_mask, torch.full_like(next_token_logits, -1e9), next_token_logits)

            filtered_probs = min_p_sampling(next_token_logits, p_base)

            if torch.isnan(filtered_probs).any():
                # print("Warning: NaN values detected in probabilities. Using uniform distribution.")
                filtered_probs = torch.ones_like(filtered_probs) / filtered_probs.shape[-1]

            try:
                next_token = torch.multinomial(filtered_probs, num_samples=1)
            except RuntimeError as e:
                print(f"Error during sampling: {e}")
                print("Falling back to argmax selection.")
                next_token = filtered_probs.argmax(dim=-1).unsqueeze(-1)

            # Append the new token to the sequence
            x = torch.cat([x, next_token], dim=1)

    output_tokens = []
    for i in range(batch_size):
        tokens = x[i, -(seq_length+1):].tolist()
        output_tokens.append(tokens)
    return output_tokens


def save_audio_files(prefill_tokens, sample_tokens, tokenizer, folder, prefix):
    if prefill_tokens is not None:
        separators = torch.tensor([4097], dtype=torch.long, device=device).unsqueeze(0).repeat(len(sample_tokens), 1)
        prefill_tokens = torch.cat([prefill_tokens, separators], dim=1)
    for i in tqdm(range(len(sample_tokens)), dynamic_ncols=True, desc="Decoding audio and saving files", position=0, leave=True):
        audio = tokenizer.decode(np.array([sample_tokens[i]]), strict=False)
        audio_np = audio.cpu().detach().numpy()
        # Normalize to [-1, 1] range
        audio_np = audio_np / np.max(np.abs(audio_np))
        # Convert to 16-bit PCM
        audio_16bit = (audio_np * 32767).astype(np.int16)
        if prefill_tokens is not None:
            prefill_audio = tokenizer.decode(np.array([prefill_tokens[i].tolist()]))
            prefill_np = prefill_audio.cpu().detach().numpy()
            prefill_np = prefill_np / np.max(np.abs(prefill_np))
            prefill_16bit = (prefill_np * 32767).astype(np.int16)
            audio_16bit = np.append(prefill_16bit, audio_16bit)
        filename = os.path.join(folder, f"{prefix}_{i}.mp3")
        write(filename, tokenizer.sample_rate, audio_16bit)
    del audio


def save_checkpoint(step, val_loss, checkpoint_name, final=False):
    checkpoint_path = os.path.join(log_dir, checkpoint_name)
    print(f"Writing checkpoint to {checkpoint_path}...", end='', flush=True)
    checkpoint = {
        'model': raw_model.state_dict(),
        'config': raw_model.config,
        'optimizers': [opt.state_dict() for opt in raw_model.optimizers],
        'schedulers': [sched.state_dict() for sched in raw_model.schedulers],
        'step': step,
        'val_loss': val_loss,
        'norms_window_adamw': norms_window_adamw,
        'norms_window_muon': norms_window_muon,
        'loss_window': loss_window,
        "divergence_window": divergence_window,
        "grok_start_step": grok_start_step,
        "tokens_trained": tokens_trained,
        "time_trained": time_trained
    }
    torch.save(checkpoint, checkpoint_path)
    print("Done.")
    if final:
        checkpoint_path = os.path.join(log_dir, f"{checkpoint_prefix}_final_model_only.pt")
        print(f"Writing final model-only checkpoint to {checkpoint_path}...", end='', flush=True)
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
        }
        torch.save(checkpoint, checkpoint_path)
        print("Done.")


eval_every = 100 #50  # Gets changed below
val_loss_steps = 50 #25  # Gets changed below
current_epoch = step * total_batch_size // train_loader.total_tokens
total_panic_adamw = 0
total_panic_muon = 0
optimizer_resets_adamw = 0
optimizer_resets_muon = 0
clip_val_adamw, clip_val_muon = get_clip_value()
skipped_steps = 0
skipped_last = False
a, b = get_lrs(0)
lr_adamw = a * max_lr_adamw
lr_muon = b * max_lr_muon

# create the log directory we will write checkpoints to and log to
os.makedirs(log_dir, exist_ok=True)

# Register pause/resume handlers.
# To pause:
#  kill -SIGUSR1 <PID>
# To resume:
#  kill -SIGUSR2 <PID>
signal.signal(signal.SIGUSR1, pause_handler)
signal.signal(signal.SIGUSR2, resume_handler)

t = tqdm(range(step, max_steps), initial=step, total=max_steps, desc=f"Training epoch {current_epoch+1} of {num_epochs}", dynamic_ncols=True)
try:
    for current_step in t:
        step = current_step
        t0 = time.time()
        last_step = (step == max_steps - 1)
        new_epoch = step * total_batch_size / train_loader.total_tokens == 0

        # once in a while evaluate our validation loss
        if step > 0 and (step % eval_every == 0 or step % save_every == 0 or last_step or new_epoch):
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x = x.to(device)
                    y = y.to(device)
                    if 'cuda' in device:
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            _, loss = model(x, y)
                    else:
                        _, loss = model(x, y)

                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                wandb.log({
                    "val/loss": val_loss_accum.item()
                }, step=step)
                if len(loss_window) > 0:
                    divergence_window.append(val_loss_accum.item() / loss_window[-1])
                if len(divergence_window) > divergence_window_size:
                    divergence_window.pop(0)

                if last_step or new_epoch or step % save_every == 0 or val_loss_accum.item() < best_val_loss:
                    best_val_loss = min(best_val_loss, val_loss_accum.item())
                    if step > warmup_steps_cosine:
                        # Probably these should be parameters
                        if best_val_loss < 4.99:
                            val_loss_steps = 70
                            eval_every = 100
                        elif best_val_loss < 5.09:
                            val_loss_steps = 50
                            eval_every = 200
                        else:
                            val_loss_steps = 24
                            eval_every = 400
                    # Don't save on the first step when resuming
                    if not resume or step != chkpt["step"]:
                        name = f"{checkpoint_prefix}_s{step:05d}_vl{val_loss_accum.item():.4f}.pt" if step % save_every == 0 else "model.pt"
                        save_checkpoint(step, best_val_loss, name, final=last_step)
                        # Sample generation
                        if step % save_every == 0 or last_step or new_epoch:
                            try:
                                print("\nGenerating audio samples...")
                                torch.cuda.empty_cache()
                                audio_folder = os.path.join(log_dir, f"{checkpoint_prefix}_s{step:05d}_vl{val_loss_accum.item():.4f}")
                                os.makedirs(audio_folder, exist_ok=True)

                                tokenizer = AudioTokenizer(device=device)

                                model.eval()
                                with torch.no_grad():
                                    try:
                                        print("\n\tFull sequence...")
                                        full_tokens = generate_tokens(model, T, inference_batch_size)
                                        # print(f"Tokens generated: {full_tokens} (len {len(full_tokens[0])}). Saving files.")
                                        print(f"Tokens generated: {len(full_tokens[0])}. Saving files.")
                                        save_audio_files(None, full_tokens, tokenizer, audio_folder, "full")
                                        del full_tokens
                                    except Exception as e:
                                        print(f"\nError generating audio sample: {e}.")
                                        print(format_exc())

                                    print("\nDone.")
                                del tokenizer
                            except Exception as e:
                                print(f"\nError generating audio samples: {e}.")
                                print(format_exc())
                            finally:
                                torch.cuda.empty_cache()

        # do one step of the optimization
        model.train()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)

            loss_accum += loss.detach() / grad_accum_steps
            loss = loss / grad_accum_steps

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()

            if grok_enabled and step >= warmup_steps_cosine * 2:
                if grok_start_step >= 0:
                    warmup_factor = min(1.0, (step - grok_start_step) / grok_warmup_steps) ** 3
                    alpha = grok_alpha * warmup_factor
                    lamb = grok_lamb  # * warmup_factor
                    grads = gradfilter_ema(model, grads=grads, alpha=alpha, lamb=lamb)
                    wandb.log({
                        "debug/grok_warmup_factor": warmup_factor,
                        "debug/grok_alpha": alpha,
                        "debug/grok_lamb": lamb,
                    }, step=step)
                else:
                    divergence = np.mean(divergence_window)
                    if divergence > grok_start_divergence:
                        print(f"Starting FastGrok (mean divergence {divergence} > {grok_start_divergence}, the grok_start_divergence; will warm up grok for {grok_warmup_steps} steps).")
                        grok_start_step = step

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        clip_val_adamw, clip_val_muon = get_clip_value()
        norm_adamw, norm_muon, skipped = model.step(clip_adamw=clip_adamw, clip_muon=clip_muon,skipped_last=skipped_last)
        if skipped:
            print(f"Skipping step {step} due to high gradient norm: {norm_adamw:.4f} (AdamW), {norm_muon:.4f} (Muon)")
            skipped_steps += 1
            wandb.log({
                "debug/skipped_step": 1,
                "debug/skipped_total": skipped_steps,
                "debug/skipped_norm_adamw": norm_adamw,
                "debug/skipped_norm_muon": norm_muon,
            }, step=step)
            skipped_last = True
        else:
            norms_window_adamw.append(norm_adamw)
            norms_window_muon.append(norm_muon)
            loss_window.append(loss_accum.item())
            if len(norms_window_adamw) > norms_window_size:
                norms_window_adamw.pop(0)
                norms_window_muon.pop(0)
                loss_window.pop(0)
            if skipped_last:
                wandb.log({
                    "debug/skipped_step": 0,
                }, step=step)
            skipped_last = False
        # get_lr is called by the schedulers during model.step(), but we also call it here to log the current learning rate
        a, b = get_lrs(step)
        lr_adamw = a * max_lr_adamw
        lr_muon = b * max_lr_muon
        if device_type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_per_sec = total_batch_size / dt
        time_trained += dt / 3600.0
        if master_process:
            prev_epoch = current_epoch
            current_epoch = step * total_batch_size // train_loader.total_tokens
            if prev_epoch != current_epoch:
                t.set_description(f"Training epoch {current_epoch+1} of {num_epochs}", refresh=True)
            tokens_trained += total_batch_size
            wandb.log({
                "etc/step": step,
                "etc/epoch": current_epoch,
                "etc/lr_adamw": lr_adamw,
                "etc/lr_muon": lr_muon,
                "etc/norm_adamw": norm_adamw,
                "etc/norm_muon": norm_muon,
                "etc/clip_value_adamw": clip_val_adamw,
                "etc/clip_value_muon": clip_val_muon,
                "etc/toks_per_sec": tokens_per_sec,
                "etc/toks_trained": tokens_trained,
                "etc/hours_trained": time_trained,
                "train/loss": loss_accum.item(),
            }, step=step)

        if pause_training:
            if master_process:
                print("Training paused. Moving model and optimizers to CPU...")

            # Ensure all processes are synchronized
            if ddp:
                dist.barrier()
                # Unwrap DDP
                model = model.module

            # Store current device
            original_device = device
            # Move model and optimizers to CPU
            model.move_optimizers_to_cpu()
            model = model.cpu()
            # Clear CUDA cache
            if 'cuda' in original_device:
                torch.cuda.empty_cache()

            if master_process:
                print("Ready and waiting for resume...")

            # Wait until resumed
            while pause_training:
                time.sleep(1)

            if master_process:
                print("Resuming training. Moving model and optimizers back to GPU...")

            # Move model and optimizers back to original device
            model = model.to(original_device)
            model.move_optimizers_to_device(original_device)
            # Rewrap in DDP if needed
            if ddp:
                model = DDP(model, device_ids=[ddp_local_rank])
                # Ensure all processes are synchronized after model is moved
                dist.barrier()

            if master_process:
                print("Model restored. Continuing training...")

except KeyboardInterrupt:
    print("\nInterrupted. Saving checkpoint...")
    save_checkpoint(step, best_val_loss, f"{checkpoint_prefix}_interrupted_s{step}.pt")

if ddp:
    destroy_process_group()
