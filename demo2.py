import json

# 加载分词器
with open("fast/tokenizer/fast_tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = json.load(f)
vocab2id = tokenizer["vocab"]
vocab = set(vocab2id.keys())

# 按照 vocab 长度从大到小排序，保证贪心优先匹配长token
sorted_vocab = sorted(vocab, key=lambda x: -len(x.split(',')))

def encode(sequence, sorted_vocab):
    tokens = sequence.strip().split()
    result = []
    i = 0
    while i < len(tokens):
        matched = False
        # 最长优先匹配
        for l in range(len(tokens), 0, -1):
            candidate = ",".join(tokens[i:i+l])
            if candidate in sorted_vocab:
                result.append(candidate)
                i += l
                matched = True
                break
        if not matched:
            # fallback: 单个token
            result.append(tokens[i])
            i += 1
    return result

# 测试序列
test_seq = "[BOS] 0 -1 -1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 [EOS]"
encoded_tokens = encode(test_seq, sorted_vocab)
encoded_ids = [vocab2id.get(tok, vocab2id.get("[UNK]", -1)) for tok in encoded_tokens]

# 打印
print("原始序列:", test_seq)
print("编码结果（token）:", encoded_tokens)
print("编码结果（id）:", encoded_ids)
