import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import os

data = np.load("data/training_pairs_original.npz")
action_chunks = data["action_chunk"]  # (N, 50)

gammas = [5, 10, 15, 20, 25, 30]

sample_chunks = action_chunks

loss_dict = {}
vocab_size_dict = {}

print("Gamma\tMean Abs Error\tStd Abs Error\tBase Vocab Size")
for gamma in gammas:
    abs_errors = []
    vocab_set = set()
    for chunk in sample_chunks:
        # DCT + 量化
        dct_coeffs = dct(chunk, norm='ortho')
        quantized = np.round(gamma * dct_coeffs).astype(int)
        # 反量化 + IDCT
        dequantized = quantized / gamma
        recovered = idct(dequantized, norm='ortho')
        # 计算平均每步绝对误差
        abs_err = np.mean(np.abs(chunk - recovered))
        abs_errors.append(abs_err)
        # 统计基本词汇
        tokens = ["[BOS]"] + list(map(str, quantized)) + ["[EOS]"]
        vocab_set.update(tokens)
    loss_dict[gamma] = abs_errors
    # 强制加入特殊token
    special_tokens = ["[BOS]", "[EOS]"]
    for tok in special_tokens:
        vocab_set.add(tok)
    vocab_size_dict[gamma] = len(vocab_set)
    print(f"{gamma}\t{np.mean(loss_dict[gamma]):.6f}\t{np.std(loss_dict[gamma]):.6f}\t{vocab_size_dict[gamma]}")

# 可视化 平均每步绝对误差 和 基本词汇量在同一张图
fig, ax1 = plt.subplots(figsize=(8, 6))

color = 'tab:blue'
ax1.set_xlabel("Gamma (Quantization Scale)")
ax1.set_ylabel("Mean Absolute Error (IDCT Reconstruction)", color=color)
means = [np.mean(loss_dict[g]) for g in gammas]
stds = [np.std(loss_dict[g]) for g in gammas]
ax1.errorbar(gammas, means, yerr=stds, fmt='-o', capsize=4, color=color, label="Mean Abs Error")
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()
color = 'tab:red'
vocab_sizes = [vocab_size_dict[g] for g in gammas]
ax2.set_ylabel("Base Vocab Size", color=color)
ax2.plot(gammas, vocab_sizes, marker='s', color=color, label="Base Vocab Size")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("Mean Absolute Error and Base Vocab Size vs. Gamma", pad=20)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # 让上方留出更多空间
plt.show()

# 0, 1, 2, 3, 4, 5, 6, 7, 
# 9, 11, 13, 15, 17, 19, 21, 23, 
# 26, 29, 32, 35, 38, 41, 44, 47, 50

# for i in range(9):
#     print(f"Running train.train_pi0, round {i+1}/9 ...")
#     os.system("python -m train.train_pi0")