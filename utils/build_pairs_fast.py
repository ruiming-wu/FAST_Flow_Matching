import numpy as np
from fast.encoder import encoder

# 1. 读取 state
print("Step 1: 加载原始 state ...")
data = np.load("data/training_pairs_original.npz")
states = data["state"]  # (N, 4)
print(f"  state 加载完成，数量: {len(states)}")

# 2. 读取 corpus 并编码
corpus_file = "fast/tokenizer/fast_tokenizer_corpus.txt"
tokenizer_path = "fast/tokenizer/fast_tokenizer.json"
token_id_seqs = []
print("Step 2: 读取语料并编码 ...")
with open(corpus_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        _, ids = encoder(line, tokenizer_path)
        token_id_seqs.append(ids)
        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1} 行")

print(f"  编码完成，总计: {len(token_id_seqs)} 行")

# 3. 检查数量是否一致
assert len(states) == len(token_id_seqs), f"states: {len(states)}, token_seqs: {len(token_id_seqs)}"

# 4. 补齐 token seq
print("Step 3: 补齐 token_seq ...")
max_len = max(len(seq) for seq in token_id_seqs)
token_id_seqs_padded = np.full((len(token_id_seqs), max_len), 2, dtype=np.int32)
for i, seq in enumerate(token_id_seqs):
    token_id_seqs_padded[i, :len(seq)] = seq
    if i < 3:
        print(f"  示例 token_seq[{i}]: {seq}")

print(f"  补齐完成，最大长度: {max_len}")

# 5. 保存
print("Step 4: 保存 npz 文件 ...")
np.savez("data/training_pairs_fast.npz", state=states, token_seq=token_id_seqs_padded)
print(f"  已保存: {states.shape[0]} 对，token_seq shape: {token_id_seqs_padded.shape}")

# 打印前10组 state 和 token seq
print("\n前10组 state 和 token_seq（已补齐）:")
for i in range(min(10, len(states))):
    print(f"state[{i}]: {states[i]}")
    print(f"token_seq[{i}]: {token_id_seqs_padded[i]}")