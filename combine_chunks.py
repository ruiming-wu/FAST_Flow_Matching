import numpy as np
import os
import glob

def main():
    # Locate all chunk .npy files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chunk_dir = os.path.join(script_dir, "data", "chunks")
    output_path = os.path.join(chunk_dir, "all_chunks.npy")

    # List all .npy files
    all_npy_files = sorted(glob.glob(os.path.join(chunk_dir, "*.npy")))

    # Define blacklist of files to skip
    blacklist = {
        "all_chunks.npy",
        "train_chunks.npy",
        "test_chunks.npy",
        "val_chunks.npy",
    }

    # Keep everything except the blacklist
    chunk_files = [
        f for f in all_npy_files
        if os.path.basename(f) not in blacklist
    ]

    if not chunk_files:
        print("[!] No valid chunk files found in data/chunks/")
        return

    print(f"Found {len(chunk_files)} chunk files.")

    all_chunks = []
    for file_path in chunk_files:
        try:
            chunk_array = np.load(file_path)
            if chunk_array.ndim != 3:
                print(f"[Skip] Unexpected shape in {file_path}: {chunk_array.shape}")
                continue
            all_chunks.append(chunk_array)
        except Exception as e:
            print(f"[Error] Failed to load {file_path}: {e}")

    if not all_chunks:
        print("No valid chunks to combine.")
        return

    combined = np.concatenate(all_chunks, axis=0)  # shape: (N, 50, 1)
    np.save(output_path, combined)

    print(f"Combined {combined.shape[0]} total chunks.")
    print(f"Saved combined array to: {output_path}")

if __name__ == "__main__":
    main()
