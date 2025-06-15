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
        except Exception as e:
            print(f"File {fname} failed to load, error message: {e}")

print("Batch check completed.")