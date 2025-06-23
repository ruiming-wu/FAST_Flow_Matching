import numpy as np

data = np.load("data/training_pairs_fast.npz")
token_seq = data["token_seq"]  # shape: (N, L)

# 统计平均长度（不计入0, 1, 2, 3）
ignore_set = {0, 1, 2, 3}
lengths = []
count_dict = {0: 0, 1: 0, 2: 0, 3: 0}

for row in token_seq:
    # 统计特殊值出现次数
    for v in count_dict:
        count_dict[v] += np.sum(row == v)
    # 统计有效长度
    valid = [x for x in row if x not in ignore_set]
    lengths.append(len(valid))

avg_len = sum(lengths) / len(lengths) if lengths else 0

print(f"token_seq 平均有效长度（不计0,1,2,3）: {avg_len:.6f}")
for v in count_dict:
    print(f"token_seq 中 {v} 出现次数: {count_dict[v]}")