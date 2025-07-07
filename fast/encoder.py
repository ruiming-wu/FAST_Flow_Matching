import json

def encoder(sequence, tokenizer_path, unk_token="[UNK]"):
    """
    Load the tokenizer and encode the input sequence, returning the token sequence and id sequence.
    :param sequence: str, space-separated token sequence
    :param tokenizer_path: str, path to the tokenizer json file
    :param unk_token: str, identifier for unknown tokens
    :return: (tokens, ids)
    """
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
    vocab2id = tokenizer["vocab"]
    vocab = set(vocab2id.keys())
    sorted_vocab = sorted(vocab, key=lambda x: -len(x.split(',')))

    tokens = []
    ids = []
    seq_tokens = sequence.strip().split()
    i = 0
    while i < len(seq_tokens):
        matched = False
        for l in range(len(seq_tokens), 0, -1):
            candidate = ",".join(seq_tokens[i:i+l])
            if candidate in sorted_vocab:
                tokens.append(candidate)
                ids.append(vocab2id.get(candidate, vocab2id.get(unk_token, 3)))
                i += l
                matched = True
                break
        if not matched:
            tokens.append(seq_tokens[i])
            ids.append(vocab2id.get(seq_tokens[i], vocab2id.get(unk_token, 3)))
            i += 1
    return tokens, ids

# Example usage
if __name__ == "__main__":
    test_seq = "[BOS] -1 1 0 1 1 2 2 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 [EOS]"
    tokens, ids = encoder(test_seq, "fast/tokenizer/fast_tokenizer.json")
    print("Encoded tokens:", tokens)
    print("Encoded ids:", ids)