import json
import matplotlib.pyplot as plt
import os

# 你可以根据需要修改参数列表
num_runs = 8
for i in range(num_runs):
    print(f"=== Running train.train_pi0_fast, round {i+1}/{num_runs} ===")
    os.system("python -m train.train_pi0_fast")