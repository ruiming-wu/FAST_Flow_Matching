import json
from collections import defaultdict
from typing import List, Tuple, Dict

corpus_path = "fast/tokenizer/fast_tokenizer_corpus.txt"
base_vocab_path = "fast/tokenizer/fast_tokenizer_base_vocab.txt"
target_vocab_size = 256  # 可根据需要调整

print("Step 1: 读取 base vocab ...")
with open(base_vocab_path, "r", encoding="utf-8") as f:
    base_vocab = set(line.strip() for line in f if line.strip())
print(f"  base vocab 加载完成，token 数量: {len(base_vocab)}")

print("Step 2: 读取 corpus ...")
corpus = []
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        tokens = line.strip().split()
        if tokens:
            corpus.append(tokens)
print(f"  corpus 加载完成，序列数: {len(corpus)}")

def get_pair_counts(corpus: List[List[str]]) -> Dict[Tuple[str, str], int]:
    pair_counts = defaultdict(int)
    for seq in corpus:
        for i in range(len(seq) - 1):
            # 跳过以 [BOS] 结尾或以 [EOS] 开头的pair
            if seq[i] == "[BOS]" or seq[i+1] == "[EOS]":
                continue
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1
    return pair_counts

def merge_most_frequent_pair(
    corpus: List[List[str]], pair_to_merge: Tuple[str, str]
) -> Tuple[List[List[str]], str]:
    a, b = pair_to_merge
    merged_token = f"{a},{b}"
    new_corpus = []
    for seq in corpus:
        new_seq = []
        skip = False
        for i in range(len(seq)):
            if skip:
                skip = False
                continue
            if i < len(seq)-1 and seq[i] == a and seq[i+1] == b:
                new_seq.append(merged_token)
                skip = True
            else:
                new_seq.append(seq[i])
        new_corpus.append(new_seq)
    return new_corpus, merged_token

def train_custom_bpe(
    corpus: List[List[str]], base_vocab: set, target_vocab_size: int
) -> Tuple[set, List[Tuple[Tuple[str, str], str]], List[List[str]]]:
    vocab = set(base_vocab)
    merges = []
    step = 0
    print("Step 3: 开始BPE训练 ...")
    while len(vocab) < target_vocab_size:
        pair_counts = get_pair_counts(corpus)
        if not pair_counts:
            print("  没有更多可合并的token对，提前结束。")
            break
        most_common = max(pair_counts.items(), key=lambda x: x[1])[0]
        corpus, merged_token = merge_most_frequent_pair(corpus, most_common)
        vocab.add(merged_token)
        merges.append((most_common, merged_token))
        step += 1
        if step % 10 == 0 or len(vocab) == target_vocab_size:
            print(f"    [进度] 当前vocab size: {len(vocab)} / {target_vocab_size}, 已合并 {step} 次")
    print("  BPE训练完成。")
    return vocab, merges, corpus

# 执行训练
vocab, merges, compressed_corpus = train_custom_bpe(corpus, base_vocab, target_vocab_size)

# 保存分词器配置
print("Step 4: 保存分词器配置 ...")
# 先保留base_vocab原顺序
with open(base_vocab_path, "r", encoding="utf-8") as f:
    base_vocab_list = [line.strip() for line in f if line.strip()]

# merge生成的新token
merged_tokens = [m[1] for m in merges if m[1] not in base_vocab_list]

# 构建token->id字典
all_tokens = base_vocab_list + merged_tokens
vocab_dict = {tok: idx for idx, tok in enumerate(all_tokens)}

tokenizer_data = {
    "vocab": vocab_dict,
    "merges": merges
}
with open("fast/tokenizer/fast_tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
print("  保存完成。")

print(f"Final Vocab Size: {len(tokenizer_data['vocab'])}")
