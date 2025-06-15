import os
import numpy as np
import matplotlib.pyplot as plt

trajs_dir = "data/trajs"

for fname in os.listdir(trajs_dir):
    if fname.endswith(".npy"):
        path = os.path.join(trajs_dir, fname)
        try:
            arr = np.load(path)
            if arr.shape != (100, 5):
                print(f"File {fname} has the shape {arr.shape}, not (100, 5)")
            else:
                last20_mean = np.mean(arr[-20:, 1])
                if abs(last20_mean) >= 0.01:
                    print(f"File {fname} has the mean value: {last20_mean:.4f} of last 20 rows, above 0.01")
        except Exception as e:
            print(f"File {fname} failed to load, error message: {e}")

print("Batch check completed.")