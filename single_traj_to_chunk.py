
def convert_npz_to_chunk(traj_path, chunk_size=50, stride=10, action_range=3.0):
    import numpy as np
    import os
    
    data = np.load(traj_path) #single_traj_to_chunk.py --traj_path data/trajs/manual_Kp8.0_Ki0.0_Kd0.0.npz 
    if 'actions' not in data or 'pid' not in data:
        print(f"[Skip] Missing 'actions' or 'pid' in {traj_path}")
        return

    actions = data['actions']  # shape (250, 1)
    actions = actions / action_range   # Normalize to [-1, 1]

    chunks = []
    for i in range(0, len(actions) - chunk_size + 1, stride):
        chunk = actions[i:i+chunk_size, :]
        if chunk.shape[0] == chunk_size:
            chunks.append(chunk)

    if not chunks:
        print(f"[Skip] No valid chunks in {traj_path}")
        return

    basename = os.path.splitext(os.path.basename(traj_path))[0]
    filename = basename + f"_chunk_stride{stride}.npy"

    output_dir = os.path.join(os.path.dirname(__file__), "data", "chunks")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"[Skip] Already exists: {filename}")
    else:
        np.save(output_path, chunks)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--action_range", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    convert_npz_to_chunk(args.traj_path, args.chunk_size, args.stride, args.action_range)
