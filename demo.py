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
        except Exception as e:
            print(f"文件 {fname} 加载失败，错误信息：{e}")

print("批量检查完成。")

