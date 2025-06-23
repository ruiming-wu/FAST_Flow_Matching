import os
import numpy as np

trajs_dir = "data/trajs"
all_states = []
all_action_chunks = []

for fname in os.listdir(trajs_dir):
    if fname.endswith(".npy"):
        arr = np.load(os.path.join(trajs_dir, fname))  # shape: (100, 5)
        states = arr[:, :4]      # (100, 4)
        actions = arr[:, 4]      # (100,)

        # list = [0, 2, 4, 6, 8, 10, 12, 14,
        #         16, 18, 20, 22, 24, 26, 28, 30,
        #         32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
        list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 
                10, 12, 14, 16, 18, 20, 22, 24, 
                27, 30, 33, 36, 39, 42, 45, 48]
        # 构造25个训练对：状态i预测动作i:i+50
        for i in list:
            state = states[i]              # (4,)
            action_chunk = actions[i:i+50] # (50,)
            if action_chunk.shape[0] == 50:
                all_states.append(state)
                all_action_chunks.append(action_chunk)

all_states = np.stack(all_states, axis=0)              # (104000, 4)
all_action_chunks = np.stack(all_action_chunks, axis=0) # (104000, 50)

np.savez("data/training_pairs_original.npz", state=all_states, action_chunk=all_action_chunks)
print(f"Saved: {all_states.shape[0]} pairs to data/training_pairs_original.npz")