import numpy as np
import os
import glob
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
chunk_dir = os.path.join(script_dir, "data", "chunks")
combined_path = os.path.join(chunk_dir, "all_chunks.npy")

# === Check a few individual chunk files ===
chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "*.npy")))
print(f"Found {len(chunk_files)} chunk files.")

for path in chunk_files[:5]:  
    data = np.load(path)
    print(f"{os.path.basename(path)}: shape={data.shape}, min={np.min(data):.3f}, max={np.max(data):.3f}")
    assert data.ndim == 3 and data.shape[1] == 50, "Each chunk should have shape (N, 50, D)"

# === Check merged chunk file ===
if os.path.exists(combined_path):
    all_chunks = np.load(combined_path)
    print(f"all_chunks.npy: shape={all_chunks.shape}, min={np.min(all_chunks):.3f}, max={np.max(all_chunks):.3f}")
    assert all_chunks.ndim == 3 and all_chunks.shape[1] == 50
else:
    print("Combined all_chunks.npy not found.")
    all_chunks = None

# === Optional: plot a few chunks ===
if all_chunks is not None:
    plt.figure(figsize=(10, 4))
    for i in range(min(5, len(all_chunks))):
        # Plot 1D action (assume D=1), shape = (50, 1)
        plt.plot(all_chunks[i].squeeze(), label=f"Chunk {i}")
    plt.title("Sample action chunks")
    plt.xlabel("Step")
    plt.ylabel("Normalized Action")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
