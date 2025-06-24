import json

def decoder(ids, tokenizer_path, skip_special_tokens=False):
    """
    将id序列解码为token（字符串），再将token按逗号分割为int，返回一个int列表
    :param ids: List[int]，token id序列
    :param tokenizer_path: str，分词器json文件路径
    :param skip_special_tokens: 是否跳过特殊符号
    :return: List[int]，解码后的int列表
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
        # 跳过特殊符号
        if token in special_tokens:
            continue
        # 逗号分割为int
        for t in token.split(","):
            t = t.strip()
            if t:  # 跳过空字符串
                try:
                    result.append(int(t))
                except ValueError:
                    pass  # 跳过无法转为int的token
    return result

# 示例用法
if __name__ == "__main__":
    ids = [0, 52, 148, 107, 143, 114, 1]
    decoded = decoder(ids, "fast/tokenizer/fast_tokenizer.json")
    print("Decoded int list:", decoded)
    print("length of decoded list:", len(decoded))