import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    token_path = "data/tokens/train_token_ids.npy"
    assert os.path.exists(token_path), f"File not found: {token_path}"

    # ==== Loading tokens ====
    token_ids = np.load(token_path, allow_pickle=True)  # ragged list
    print(f"Loaded {len(token_ids)} token sequences.")

    # ==== Check first 3 tokens ====
    print("\n🔎 First 3 token sequences:")
    for i in range(min(3, len(token_ids))):
        print(f"Sequence {i}: length = {len(token_ids[i])}, tokens = {token_ids[i][:10]}...")

    # ==== length distribution ====
    lengths = [len(seq) for seq in token_ids]
    print(f"\n📊 Token length stats:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths)}")

    # ==== Visualization ====
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title("Token Sequence Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
