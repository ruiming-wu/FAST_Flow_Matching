import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import dct
from processing_action_tokenizer import UniversalActionProcessor

def main():
    # === Config ===
    chunk_path = "data/chunks/test_chunks.npy"
    vocab_size = 1024
    scale = 10

    # === Load chunk data ===
    assert os.path.exists(chunk_path), f"Missing: {chunk_path}"
    chunks = np.load(chunk_path)  # (N, L, D)
    N, L, D = chunks.shape
    print(f"Loaded chunks: shape = ({N}, {L}, {D})")

    # === Train tokenizer temporarily (in RAM) ===
    print("Fitting temporary tokenizer in memory...")
    processor = UniversalActionProcessor.fit(
        action_data=chunks,
        vocab_size=vocab_size,
        scale=scale,
        time_horizon=L,
        action_dim=D
    )
    tokenizer = processor.bpe_tokenizer

    # === Encode chunks one by one ===
    all_tokens = []
    token_lengths = []

    for i in range(N):
        chunk = chunks[i]
        dct_chunk = dct(chunk, type=2, norm='ortho', axis=0)
        dct_scaled = np.round(dct_chunk * scale).astype(int).flatten()
        text = " ".join(map(str, dct_scaled))
        token_ids = tokenizer.encode(text)
        all_tokens.extend(token_ids)
        token_lengths.append(len(token_ids))

    # === Stats ===
    print(f"\nToken usage stats:")
    print(f"  Total tokens used: {len(all_tokens)}")
    print(f"  Unique tokens used: {len(np.unique(all_tokens))} / {vocab_size}")
    print(f"  Avg token length per chunk: {np.mean(token_lengths):.2f}")
    print(f"  Median: {np.median(token_lengths)}, Max: {np.max(token_lengths)}, Min: {np.min(token_lengths)}")

    # === Plot token distribution ===
    plt.hist(all_tokens, bins=100)
    plt.title("Token ID Distribution")
    plt.xlabel("Token ID")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
