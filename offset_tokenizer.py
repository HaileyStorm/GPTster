import torch, torchaudio
from snac import SNAC
import numpy as np


class AudioTokenizer:
    """
    A class for tokenizing and detokenizing audio data using the SNAC model.

    This tokenizer converts audio waveforms into a flattened, hierarchical representation
    and vice versa. It uses a pre-trained SNAC model to encode audio into multiple tensors
    of varying granularity, which are then flattened into a single sequence of integers.

    Input Structure:
        - A list of audio waveforms (torch tensors) with a sample rate of 24/32/44kHz (depending on selected SNAC model).

    Output Structure:
        A flattened sequence of integers with the following structure:
        [A1, B1, B2, C1, C2, C3, C4, D1, D2, D3, D4, D5, D6, D7, D8, A2, B3, B4, C5, C6, C7, C8, D9, D10, D11, D12, D13, D14, D15, D16, ..., A144, B287, B288, ...]

        Where:
        - A, B, C, and D represent values from the four SNAC tensors respectively, with offsets applied:
          - A (tensor 0): 0-4095
          - B (tensor 1): 4099-8194
          - C (tensor 2): 8196-12291
          - D (tensor 3, when present): 12293-16388
        - The A, B, C and D tensors for an entire sequence are called "codes." We refer to a 7/15-token, that is, either
        ABBCCCC or ABBCCCCDDDDDDDD, as a "code group."

    Vocabulary sizes:
        - 24kHz model: 12292 tokens
        - 32kHz and 44kHz models: 16389 tokens

    The flattened structure preserves the hierarchical relationship between the tensors:
    - For each timestep in the coarsest tensor (A):
        - One value from the first (coarsest) tensor (A)
        - Two values from the second tensor (B)
        - Four values from the third tensor (C)
        - Eight values from the fourth (finest) tensor (D) (for 32kHz and 44kHz models)

    This structure allows for efficient encoding and decoding of audio data while
    maintaining the multi-scale representation provided by the SNAC model.

    Methods:
        encode: Converts audio waveforms to the flattened token representation.
        decode: Reconstructs audio waveforms from the flattened token representation.
    """

    def __init__(self, device = 'cpu') -> None:
        #self.model = torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().to(device))
        self.model = torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device))
        #self.sample_rate = 32000
        self.sample_rate = 44000
        self.device = device
        self.offsets = [0, 4099, 8196, 12293]

    def flatten_tensors(self, tensors):
        flattened = []
        for batch in range(tensors[0].size(0)):
            flattened_list = []
            # Iterate by code group (each tensor A token starts a code group)
            for i in range(tensors[0].size()[1]):
                for t, tensor in enumerate(tensors):
                    indices = [i * (2 ** t) + j for j in range(2 ** t)]
                    flattened_list.extend(tensor[batch][indices].add(self.offsets[t]).tolist())
            flattened.append(flattened_list)
        #print(flattened)
        return flattened

    def reconstruct_single_tensors(self, flattened_output):
        #print(flattened_output)
        num_tensors = 4 if max(flattened_output) > 12292 else 3
        tensor_lengths = [1, 2, 4, 8][:num_tensors]
        code_group_length = sum(tensor_lengths)
        assert len(flattened_output) % code_group_length == 0, f"Input length {len(flattened_output)} is not divisible by code group length {code_group_length}; likely the model is under-trained (though, now that separators aren't a thing this shouldn't happen)."

        reconstructed = [[] for _ in range(num_tensors)]

        # Rebuild the separate (full sequence) codes from the flattened code groups.
        for i in range(0, len(flattened_output), code_group_length):
            code_group = flattened_output[i:i + code_group_length]
            start = 0
            # Pull out the code tokens by iterating over the list of number of sequential tokens from each code (t gives
            # us the tensor number, 0 for A ... 3 for D.
            # We remove the offset values, which are dependent on which tensor they come from / are going to, as we go.
            for t, length in enumerate(tensor_lengths):
                for token in code_group[start:start + length]:
                    reconstructed[t].append(token - self.offsets[t])
                start += length

        return [torch.tensor(tensor).unsqueeze(0) for tensor in reconstructed]

    # expects list of waveforms formatted in 24/32/44khz mono (depending on SNAC model selection)
    def encode(self, waves):

        audio = torch.stack(waves).to(self.device)

        with torch.inference_mode():
            # Each  code is a time step, e.g. if 6 seconds audio is passed in using 32khz model you'll get 64 codes each representing ~93.75ms (3000 samples) of audio
            codes = self.model.encode(audio)
        #print(f"encode model output (`codes`) shape: {[code.shape for code in codes]}")
            
        #print("Number of tensors:", len(codes))
        #mx = 0
        #for i, code in enumerate(codes):
        #    print(f"\tTensor {i} shape: {code.shape}, min: {torch.min(code)}, max: {torch.max(code)}")
        #    mx = max(torch.max(code), mx)
        #print(f"Max value: {mx}")
        
        del audio

        with torch.no_grad():
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
        return np.array(self.flatten_tensors(codes))
    
    # of (1, T)
    def decode(self, tokens, strict=True):
        all_audio = []
        for chunk in tokens:
            raw = self.reconstruct_single_tensors(chunk)

            # Check for invalid values, raise exception or print warning & clamp, depending on `strict`
            clamp = False
            for j, tensor in enumerate(raw):
                # Flatten the tensor to 1D for accurate linear index calculations
                flat_tensor = tensor.flatten()

                # Create masks on the flattened tensor
                neg_mask = flat_tensor < 0
                pos_mask = flat_tensor > 4095

                # Count invalid values
                neg_ct = neg_mask.sum().item()
                pos_ct = pos_mask.sum().item()

                # Initialize index variables
                first_neg_idx = first_pos_idx = None
                avg_neg_idx = avg_pos_idx = None

                # Debugging: Print tensor shape and counts
                #print(f"Tensor {j} shape: {tensor.shape}")
                #print(f"Negative count: {neg_ct}, Positive count: {pos_ct}")

                # Calculate indices if invalid values exist
                if neg_ct > 0:
                    neg_indices = torch.nonzero(neg_mask, as_tuple=True)[0]
                    first_neg_idx = neg_indices[0].item()
                    avg_neg_idx = neg_indices.float().mean().item()
                    #print(f"Negative indices: {neg_indices.tolist()}")

                if pos_ct > 0:
                    pos_indices = torch.nonzero(pos_mask, as_tuple=True)[0]
                    first_pos_idx = pos_indices[0].item()
                    avg_pos_idx = pos_indices.float().mean().item()
                    #print(f"Positive indices: {pos_indices.tolist()}")

                if neg_ct > 0 or pos_ct > 0:
                    # Start constructing the message
                    message_parts = []

                    if neg_ct > 0:
                        neg_part = (
                            f"{neg_ct} negative value{'s' if neg_ct != 1 else ''} "
                            f"(First at index {first_neg_idx}, Average index {avg_neg_idx:.2f})"
                        )
                        message_parts.append(neg_part)

                    if pos_ct > 0:
                        pos_part = (
                            f"{pos_ct} value{'s' if pos_ct != 1 else ''} over 4095 "
                            f"(First at index {first_pos_idx}, Average index {avg_pos_idx:.2f})"
                        )
                        message_parts.append(pos_part)

                    # Join parts with ' and ' if both are present
                    message = " and ".join(message_parts)

                    # Complete the message with tensor index
                    full_message = (
                        f"Found {message} in tensor {j}. "
                        "This indicates invalid offsets in the input data; likely the model is under-trained."
                    )

                    if strict:
                        raise ValueError(full_message)
                    else:
                        clamp = True
                        full_message += " Strict mode is off; values will be clamped to 0-4095."
                        print(full_message)

            if clamp:
                # Clamp all tensors outside the loop to ensure it's done once
                raw = [torch.clamp(tensor, min=0, max=4095) for tensor in raw]

            codes = [tensor.to(self.device) for tensor in raw]
            #print("Reconstructed codes:")
            #for i, code in enumerate(codes):
            #    print(f'Tensor {i} shape: {code.shape}')
            #    print(f'Tensor {i} content: {code[0][:18].tolist()}..{code[0][-18:].tolist()}')
            #    print(f'Tensor {i} min: {torch.min(code)}, max: {torch.max(code)}')
            #    print()

            with torch.inference_mode():
                audio_hat = self.model.decode(codes)

            all_audio.append(audio_hat)

            del codes

            with torch.no_grad():
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()

        # Concatenate all audio chunks
        full_audio = torch.cat(all_audio, dim=-1)

        return full_audio

