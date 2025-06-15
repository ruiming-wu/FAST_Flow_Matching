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
                print(f"文件 {fname} 形状为 {arr.shape}，不是 (100, 5)")
            else:
                last20_mean = np.mean(arr[-20:, 1])
                if abs(last20_mean) >= 0.01:
                    print(f"文件 {fname} 后20行第2列均值为 {last20_mean:.4f}，绝对值不小于0.01")
        except Exception as e:
            print(f"文件 {fname} 加载失败，错误信息：{e}")

print("批量检查完成。")

