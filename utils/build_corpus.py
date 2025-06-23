import numpy as np
from scipy.fftpack import dct

GAMMA = 10

data = np.load("data/training_pairs_original.npz")
action_chunks = data["action_chunk"]  # (N, 50)

special_tokens = ["[BOS]", "[EOS]"]
corpus_path = "fast/tokenizer/fast_tokenizer_corpus.txt"
vocab_path = "fast/tokenizer/fast_tokenizer_base_vocab.txt"

def process_chunk(chunk, gamma):
    dct_coeffs = dct(chunk, norm='ortho')
    quantized = np.round(gamma * dct_coeffs).astype(int)
    return quantized

vocab_set = set()

with open(corpus_path, "w", encoding='utf-8') as f:
    for chunk in action_chunks:
        seq = process_chunk(chunk, GAMMA)
        tokens = ["[BOS]"] + list(map(str, seq)) + ["[EOS]"]
        line = " ".join(tokens)
        f.write(line + "\n")
        vocab_set.update(tokens)

# 强制加入特殊token
for tok in special_tokens:
    vocab_set.add(tok)

# 排序（特殊token优先，其余按数字排序）
vocab = [tok for tok in special_tokens]
vocab += sorted([tok for tok in vocab_set if tok not in special_tokens],
                key=lambda x: (not x.lstrip('-').isdigit(), int(x) if x.lstrip('-').isdigit() else x))

with open(vocab_path, 'w', encoding='utf-8') as f:
    for token in vocab:
        f.write(token + '\n')

print(f"Base vocab size: {len(vocab)}")