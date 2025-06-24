import numpy as np
import torch
import matplotlib.pyplot as plt
from model.transformer_pi0_fast import TransformerPi0FAST
from fast.decoder import decoder
from scipy.fftpack import idct
from utils.build_corpus import GAMMA

def run_pi0fast_inference_on_chunk(
    chunk_idx,
    model_path,
    tokenizer_path,
    original_npz_path="data/training_pairs_original.npz",
    max_seq_len=25,
    device="cpu"
):
    # 1. 加载原始数据
    data = np.load(original_npz_path)
    states = data["state"]  # (N, 4)
    actions = data["action_chunk"]  # (N, 50)
    chunk_state = states[chunk_idx]
    chunk_action = actions[chunk_idx]

    # 2. 加载模型
    state_dim = 4
    vocab_size = 256
    embed_dim = 128
    pad_id = 2
    model = TransformerPi0FAST(
        state_dim=state_dim,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        padding_id=pad_id
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. 推理token序列
    bos_id, eos_id = 0, 1
    tokens = [bos_id]
    state_tensor = torch.tensor(chunk_state, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(max_seq_len - 1):
        input_seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = (input_seq != pad_id).long()
        with torch.no_grad():
            logits = model(state_tensor, input_seq, attn_mask)
            next_token = logits[0, -1].argmax(-1).item()
        tokens.append(next_token)
        if next_token == eos_id:
            break

    # 4. 解码为int序列
    token_ints = decoder(tokens, tokenizer_path)
    # 补齐到50维
    if len(token_ints) < 50:
        token_ints = token_ints + [0] * (50 - len(token_ints))
    token_ints = token_ints[:50]

    # 5. 逆量化
    quantized = np.array(token_ints) /  GAMMA

    # 6. 逆DCT
    pred_action = idct(quantized, norm='ortho')[:50]

    # 7. 可视化
    plt.figure(figsize=(12, 5))
    plt.plot(chunk_action, label="Original Action")
    plt.plot(pred_action, label="Predicted Action (pi0fast)", linestyle='--')
    plt.title(f"Chunk {chunk_idx}: Original vs. pi0fast Predicted Action")
    plt.xlabel("Action Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("原始动作:", chunk_action)
    print("pi0fast逆推动作:", pred_action)

if __name__ == "__main__":
    # 参数请根据实际情况填写
    chunk_idx = 51  # 指定chunk编号
    model_path = "train/trained_models/tinypi0fast_20250624_1631.pth"
    tokenizer_path = "fast/tokenizer/fast_tokenizer.json"
    run_pi0fast_inference_on_chunk(chunk_idx, model_path, tokenizer_path)