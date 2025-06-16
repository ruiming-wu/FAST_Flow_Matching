import os
import numpy as np
import matplotlib.pyplot as plt

trajs_dir = "data/trajs"
count = 0
wrong_shape_count = 0
over_threshold_count = 0

for fname in os.listdir(trajs_dir):
    if fname.endswith(".npy"):
        path = os.path.join(trajs_dir, fname)
        count += 1
        try:
            arr = np.load(path)
            if arr.shape != (100, 5):
                print(f"File {fname} has the shape {arr.shape}, not (100, 5)")
                wrong_shape_count += 1
            else:
                last25_mean = np.mean(arr[-25:, 1])
                if abs(last25_mean) >= 0.01:
                    print(f"File {fname} has the mean value: {last25_mean:.4f} of last 25 rows, above 0.01")
                    over_threshold_count += 1
        except Exception as e:
            print(f"File {fname} failed to load, error message: {e}")

print("Batch check completed.")
print(f"Total files checked: {count}")
print(f"Files with wrong shape: {wrong_shape_count}")
print(f"Files with mean value over threshold: {over_threshold_count}")