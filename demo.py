import os
import numpy as np
from tqdm import tqdm


traj_dir = "data/trajs"
save_path = "data/chunks/new_all_chunks.npy"
chunk_len = 50

all_chunks = []

for fname in tqdm(os.listdir(traj_dir)):
    if not fname.endswith(".npz"):
        continue
    data = np.load(os.path.join(traj_dir, fname))
    obs = data["observations"]    # (T, 4)
    acts = data["actions"]        # (T, 1)
    T = len(obs)

    # 用每一步状态预测后面50步动作
    for t in range(T - chunk_len):
        state = obs[t]  # (4,)
        actions = acts[t+1:t+1+chunk_len]  # (50, 1)
        if len(actions) < chunk_len:
            continue
        # 拼成 (1+50, 5)：第1行为state+0，后50行为0+action
        chunk = np.zeros((chunk_len+1, 5), dtype=np.float32)
        chunk[0, :4] = state
        chunk[1:, 4] = actions[:, 0]
        all_chunks.append(chunk)

all_chunks = np.stack(all_chunks, axis=0)  # (N_chunk, 51, 5)
print("all_chunks shape:", all_chunks.shape)
print("first chunk:\n", all_chunks[0])
np.save(save_path, all_chunks)
print(f"Saved to {save_path}")
