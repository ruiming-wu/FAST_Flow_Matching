import numpy as np
from sklearn.model_selection import train_test_split
import os

def main():
    input_path = "data/chunks/all_chunks.npy"
    output_dir = "data/chunks"
    
    # Load the full chunk dataset
    assert os.path.exists(input_path), f"File not found: {input_path}"
    print(f"Loading all chunks from: {input_path}")
    all_chunks = np.load(input_path)
    N = len(all_chunks)
    print(f"Total chunks: {N}")

    # Step 1: Split out the test set (10%)
    train_val_chunks, test_chunks = train_test_split(
        all_chunks, test_size=0.1, random_state=42
    )

    # Step 2: Split the remaining 90% into train (80%) and validation (10%)
    # 1/9 ≈ 10% of the remaining 90%
    train_chunks, val_chunks = train_test_split(
        train_val_chunks, test_size=1/9, random_state=42
    )

    # Print resulting sizes
    print(f"Train chunks: {len(train_chunks)}")
    print(f"Validation chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")

    # Save the results to disk
    np.save(os.path.join(output_dir, "train_chunks.npy"), train_chunks)
    np.save(os.path.join(output_dir, "val_chunks.npy"), val_chunks)
    np.save(os.path.join(output_dir, "test_chunks.npy"), test_chunks)
    print("Saved split datasets successfully.")

if __name__ == "__main__":
    main()
