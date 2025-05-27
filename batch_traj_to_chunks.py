
import os
import glob
from single_traj_to_chunk import convert_npz_to_chunk

# === Main logic ===
if __name__ == "__main__":
    # Locate all .npz files in data/trajs/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    traj_dir = os.path.join(script_dir, "data", "trajs")
    all_npz = sorted(glob.glob(os.path.join(traj_dir, "*.npz")))

    if not all_npz:
        print("No .npz files found.")
    else:
        print(f"Found {len(all_npz)} .npz files.")
        for path in all_npz:
            convert_npz_to_chunk(path)
