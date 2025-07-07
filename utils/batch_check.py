import os
import numpy as np
import matplotlib.pyplot as plt

def get_converge_step(arr, col=1, threshold=0.02, min_len=10):
    for i in range(len(arr) - min_len + 1):
        window = arr[i:i+min_len, col] + 0.00166666
        if np.all(np.abs(window) < threshold):
            return i
    return len(arr)

trajs_dir = "data/trajs"
count = 0
wrong_shape_count = 0
over_threshold_count = 0
converge_steps = []

for fname in os.listdir(trajs_dir):
    if fname.endswith(".npy"):
        path = os.path.join(trajs_dir, fname)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} files...")
        try:
            arr = np.load(path)
            if arr.shape != (100, 5):
                print(f"File {fname} has the shape {arr.shape}, not (100, 5)")
                wrong_shape_count += 1
            else:
                step = get_converge_step(arr)
                if step == 100:
                    print(f"File {fname} not converged, step {step} >= 90")
                    over_threshold_count += 1
                converge_steps.append(step)
        except Exception as e:
            print(f"File {fname} failed to load, error message: {e}")

avg_converge_step = np.mean(converge_steps) if converge_steps else 0

print("Batch check completed.")
print(f"Total files checked: {count}")
print(f"Files with wrong shape: {wrong_shape_count}")
print(f"Files not converged: {over_threshold_count}")
print(f"Average converge step: {avg_converge_step:.2f}")