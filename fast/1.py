import numpy as np
from scipy.fftpack import dct
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

gamma = 10
data = np.load("data/training_pairs_original.npz")
action_chunks = data["action_chunk"]  # (104000, 50)

def process_chunk(chunk, gamma):
    dct_coeffs = dct(chunk, norm='ortho')
    quantized = np.round(gamma * dct_coeffs).astype(int)
    return quantized

with open("fast/tokenizer/fast_tokenizer_corpus.txt", "w") as f:
    for chunk in action_chunks:
        seq = process_chunk(chunk, gamma)
        # 转为字符串，每个token用空格分隔
        f.write(" ".join(map(str, seq)) + "\n")

# 初始化BPE分词器
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 训练
trainer = trainers.BpeTrainer(
    vocab_size=256,
    min_frequency=1,
    special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
)
tokenizer.train(["fast/tokenizer/fast_tokenizer_corpus.txt"], trainer)

# 保存分词器
tokenizer.save("fast/tokenizer/fast_tokenizer.json")

tokenizer = Tokenizer.from_file("fast/tokenizer/fast_tokenizer.json")

# 假设你有一个新的动作chunk
chunk = np.random.randn(50)
seq = process_chunk(chunk, gamma)
seq_str = " ".join(map(str, seq))
encoded = tokenizer.encode(seq_str)

print("Token ids:", encoded.ids)