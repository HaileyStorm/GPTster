from gpt2_alibi import GPT, GPTConfig
import torch
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from offset_tokenizer import AudioTokenizer
from scipy.io.wavfile import write
import math
import random
import sys
import os

checkpoint_path = './log_alibi_11_14_672_Muon_0.535x_AdamWCosine/model.pt'
output_dir = './log_alibi_11_14_672_Muon_0.535x_AdamWCosine/model_s24985_vl5.5402'
os.makedirs(output_dir, exist_ok=True)

batch_size = 2
num_batches = 15
# Recommend 0.045-0.055
p_base = 0.0515
# Recommend 0.965-0.99 (for p_base 0.045), but can get fun results up to 1.1 or more
min_p_temp = 0.955 #0.968
# Generally OK ~0.9-1.01, depending on what you're after, best ~0.935-0.975, default 0.96
top_k_temp = 0.96
# Best 640-720, default 712
top_k_max = 712
# OK ~256-512, best ~350-385, default 360
top_k_min = 360
# Number of generated tokens to take to decrease from top_k_max to top_k_min
# Depending on min/max, OK 512 on down to 128 or less, best ~384-416, default 408
top_k_warmup = 408
sampling_methods = ['min_p', 'top_k']  # ['top_k', 'min_p']
random.seed()
seed = random.randint(0, sys.maxsize)

# 3s @ 32khz = 512 tokens
# 4.5s = 768 tokens
# 6s = 1024
max_length = 2160  # 6480, 2160...

torch.set_float32_matmul_precision('high')
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

chkpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
config: GPTConfig = chkpt["config"]

model = GPT(config)

# Load model state
model_state_dict = OrderedDict([
    (key.replace('_orig_mod.', ''), value) for key, value in chkpt['model'].items()
])
model.load_state_dict(model_state_dict)

model.to(device)
model = torch.compile(model)

model.eval()


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


def get_top_k(step):
    if step < top_k_warmup:
        progress = step / top_k_warmup
        # Use a sigmoid function for a gradual start and more rapid finish
        sigmoid_progress = 1 / (1 + math.exp(-10 * (progress - 0.5)))
        return int(top_k_max - sigmoid_progress * (top_k_max - top_k_min))
    else:
        return top_k_min


def normalize_audio(audio):
    #audio = audio.squeeze()  # Remove any extra dimensions
    #audio = (audio - np.min(audio)) / (np.max(audio) - np.min(audio))  # Normalize to 0-1
    #audio = (audio * 2) - 1  # Scale to -1 to 1
    #audio = (audio * 32767).astype(np.int16)  # Scale to 16-bit integer range
    #return audio
    audio = audio / np.max(np.abs(audio))
    # Convert to 16-bit PCM
    return (audio * 32767).astype(np.int16)


for b in range(num_batches):
    # Set the seed for this batch
    batch_seed = seed + b
    torch.manual_seed(batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(batch_seed)
    random.seed(batch_seed)

    initial_tokens = [random.randint(0, 4095)]
    initial_tokens = torch.tensor(initial_tokens, dtype=torch.long)
    initial_tokens = initial_tokens.unsqueeze(0).repeat(batch_size, 1)

    for sampling_method in sampling_methods:
        torch.manual_seed(batch_seed)  # Reset seed for each method
        if torch.cuda.is_available():
            torch.cuda.manual_seed(batch_seed)

        xgen = initial_tokens.clone().to(device)
        output_tokens = []

        # Initialize the progress bar
        with tqdm(total=max_length, desc=f"Generating ({sampling_method})...") as pbar:
            # Set initial progress
            pbar.update(xgen.size(1))

            while xgen.size(1) <= max_length:
                with torch.no_grad():
                    # Get logits from the model
                    with torch.autocast(device_type=xgen.device.type, dtype=torch.bfloat16):
                        logits, _ = model(xgen)
                    next_token_logits = logits[:, -1, :]

                    # Apply temperature (use appropriate temperature for each method)
                    temp = min_p_temp if sampling_method == 'min_p' else top_k_temp
                    next_token_logits = next_token_logits / temp

                    # Handle NaN and Inf values in logits
                    nan_mask = torch.isnan(next_token_logits) | torch.isinf(next_token_logits)
                    if nan_mask.any():
                        next_token_logits = torch.where(nan_mask, torch.full_like(next_token_logits, -1e9),
                                                        next_token_logits)

                    if sampling_method == 'min_p':
                        # Min-p sampling
                        filtered_probs = min_p_sampling(next_token_logits, p_base)

                        if torch.isnan(filtered_probs).any():
                            #print("Warning: NaN values detected in probabilities. Using uniform distribution.")
                            filtered_probs = torch.ones_like(filtered_probs) / filtered_probs.shape[-1]

                        try:
                            next_token = torch.multinomial(filtered_probs, num_samples=1)
                        except RuntimeError as e:
                            print(f"Error during sampling: {e}")
                            print("Falling back to argmax selection.")
                            next_token = filtered_probs.argmax(dim=-1).unsqueeze(-1)

                    else:
                        # Top-k sampling
                        current_step = min(xgen.size(1) - 1, top_k_warmup)
                        top_k = get_top_k(current_step)
                        pbar.set_description(f"Generating (top_k={top_k})...", refresh=True)

                        probs = F.softmax(next_token_logits, dim=-1)
                        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

                        if torch.isnan(top_k_probs).any():
                            top_k_probs = torch.ones_like(top_k_probs) / top_k

                        try:
                            sample_indices = torch.multinomial(top_k_probs, num_samples=1)
                            next_token = torch.gather(top_k_indices, -1, sample_indices)
                        except RuntimeError as e:
                            print(f"Error during sampling: {e}")
                            print("Falling back to argmax selection from top-k.")
                            next_token = top_k_indices[:, 0].unsqueeze(-1)

                    # Append the new token to the sequence
                    xgen = torch.cat([xgen, next_token], dim=1)

                    pbar.update(1)  # Update by the number of new tokens added

            for i in range(batch_size):
                tokens = xgen[i, :max_length].tolist()
                output_tokens.append(tokens)

        with torch.no_grad():
            del xgen
            torch.cuda.empty_cache()

        tokenizer = AudioTokenizer(device=device)

        for i in range(batch_size):
            print(f'output_tokens ({len(output_tokens[i])}): {np.array([output_tokens[i]])}')
            try:
                audio_out = tokenizer.decode(np.array([output_tokens[i]]), strict=False)
                audio_out = normalize_audio(audio_out.cpu().detach().numpy())
                filename = os.path.join(output_dir, f"{sampling_method}_b{b}_{i}.wav")
                write(filename, tokenizer.sample_rate, audio_out)
            except Exception as e:
                print(f"Error decoding: {e}")

        with torch.no_grad():
            del output_tokens
            del tokenizer
            torch.cuda.empty_cache()