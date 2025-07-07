import numpy as np
from fast.encoder import encoder

# 1. Load original states
print("Step 1: Loading original states ...")
data = np.load("data/training_pairs_original.npz")
states = data["state"]  # (N, 4)
print(f"  States loaded, count: {len(states)}")

# 2. Read corpus and encode
corpus_file = "fast/tokenizer/fast_tokenizer_corpus.txt"
tokenizer_path = "fast/tokenizer/fast_tokenizer.json"
token_id_seqs = []
print("Step 2: Reading corpus and encoding ...")
with open(corpus_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        _, ids = encoder(line, tokenizer_path)
        token_id_seqs.append(ids)
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} lines")

print(f"  Encoding finished, total: {len(token_id_seqs)} lines")

# 3. Check if counts match
assert len(states) == len(token_id_seqs), f"states: {len(states)}, token_seqs: {len(token_id_seqs)}"

# 4. Pad token sequences
print("Step 3: Padding token sequences ...")
max_len = max(len(seq) for seq in token_id_seqs)
token_id_seqs_padded = np.full((len(token_id_seqs), max_len), 2, dtype=np.int32)
for i, seq in enumerate(token_id_seqs):
    token_id_seqs_padded[i, :len(seq)] = seq
    if i < 3:
        print(f"  Example token_seq[{i}]: {seq}")

print(f"  Padding finished, max length: {max_len}")

# 5. Save
print("Step 4: Saving npz file ...")
np.savez("data/training_pairs_fast.npz", state=states, token_seq=token_id_seqs_padded)
print(f"  Saved: {states.shape[0]} pairs, token_seq shape: {token_id_seqs_padded.shape}")

# Print first 10 state and token_seq pairs
print("\nFirst 10 state and token_seq (padded):")
for i in range(min(10, len(states))):
    print(f"state[{i}]: {states[i]}")
    print(f"token_seq[{i}]: {token_id_seqs_padded[i]}")