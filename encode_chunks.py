import numpy as np
import os
from processing_action_tokenizer import UniversalActionProcessor
from transformers import PreTrainedTokenizerFast

def main():
    chunk_path = "data/chunks/test_chunks.npy" #   artificially modify it to train/val/set
    config_path = "tokenizer_trained/config.npy"
    bpe_tokenizer_path = "tokenizer_trained/bpe_tokenizer"
    output_path = "data/tokens/test_token_ids.npy" #   modify it as above 

    # Load chunk data
    assert os.path.exists(chunk_path), f"Chunk file not found: {chunk_path}"
    action_chunks = np.load(chunk_path)
    N, L, D = action_chunks.shape
    print(f"Loaded chunk data shape: (N={N}, L={L}, D={D})")

    # Load tokenizer config
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    vocab_size, scale, min_token, time_horizon, action_dim = np.load(config_path)
    print(f"Loaded tokenizer config: vocab={vocab_size}, scale={scale}, min_token={min_token}")

    # Load bpe_tokenizer
    bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained(bpe_tokenizer_path)
    print("Loaded bpe_tokenizer from disk.")

    # Reconstruct processor with loaded tokenizer
    processor = UniversalActionProcessor(
        bpe_tokenizer=bpe_tokenizer,
        scale=scale,
        vocab_size=vocab_size,
        min_token=int(min_token),
        time_horizon=int(time_horizon),
        action_dim=int(action_dim),
    )

    # Encode chunks
    print("Encoding chunks...")
    token_ids = processor(action_chunks)

    # Save encoded tokens
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(token_ids, dtype=object))
    print(f"Saved encoded tokens to: {output_path}, total sequences: {len(token_ids)}")

if __name__ == "__main__":
    main()
