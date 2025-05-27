import numpy as np
import os
from processing_action_tokenizer import UniversalActionProcessor

def main():
    chunk_path = "data/chunks/train_chunks.npy"
    save_dir = "tokenizer_trained"
    vocab_size = 1024
    scale = 10

    assert os.path.exists(chunk_path), f"File not found: {chunk_path}"
    action_chunks = np.load(chunk_path)
    N, L, D = action_chunks.shape

    print("Training tokenizer...")
    processor = UniversalActionProcessor.fit(
        action_data=action_chunks,
        vocab_size=vocab_size,
        scale=scale,
        time_horizon=L,
        action_dim=D,
    )
    print("Training complete.")

    # Save processor config params
    os.makedirs(save_dir, exist_ok=True)
    config_array = np.array([vocab_size, scale, processor.min_token, L, D])
    np.save(os.path.join(save_dir, "config.npy"), config_array)
    
    # Save bpe_tokenizer with Hugging Face save_pretrained
    processor.bpe_tokenizer.save_pretrained(os.path.join(save_dir, "bpe_tokenizer"))
    print(f"Tokenizer saved to {save_dir}")
    processor.bpe_tokenizer.save_model("tokenizer_trained/bpe_tokenizer")
    import json
    json.dump({
        "vocab_size": vocab_size,
        "scale": scale,
        "min_token": processor.min_token,
        "time_horizon": L,
        "action_dim": D
    }, open("tokenizer_trained/config.json", "w"))
if __name__ == "__main__":
    main()
