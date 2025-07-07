import json

def decoder(ids, tokenizer_path, skip_special_tokens=False):
    """
    Decode a sequence of token ids to tokens (strings), then split tokens by comma and convert to int, returning an int list.
    :param ids: List[int], token id sequence
    :param tokenizer_path: str, path to tokenizer json file
    :param skip_special_tokens: whether to skip special tokens
    :return: List[int], decoded int list
    """
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]
    id2token = {int(v): k for k, v in vocab.items()}
    special_tokens = {"[BOS]", "[EOS]", "[PAD]", "[UNK]"}
    result = []
    for idx in ids:
        token = id2token.get(idx, "[UNK]")
        if skip_special_tokens and token in special_tokens:
            continue
        if token in special_tokens:
            continue
        for t in token.split(","):
            t = t.strip()
            if t:
                try:
                    result.append(int(t))
                except ValueError:
                    pass
    return result

# Example usage
if __name__ == "__main__":
    ids = [0, 52, 148, 107, 143, 114, 1, 2, 2, 2, 2]
    decoded = decoder(ids, "fast/tokenizer/fast_tokenizer.json")
    print("Decoded int list:", decoded)
    print("length of decoded list:", len(decoded))