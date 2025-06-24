import torch
import numpy as np
from model.transformer_pi0_fast import TransformerPi0FAST

def infer_pi0fast_token_sequence(model_path, state_vec, max_seq_len=25, device='cpu', bos_id=0, eos_id=1, pad_id=2):
    """
    推理pi0fast模型，生成token序列（遇到EOS或到达最大长度即停止）
    Args:
        model_path: 训练好的模型权重路径
        state_vec: 状态向量 (4,)
        max_seq_len: 最大生成长度
        device: 'cpu' or 'cuda'
        bos_id, eos_id, pad_id: 特殊token id
    Returns:
        List[int]: 生成的token序列（含BOS和EOS）
    """
    # 超参数（需与训练一致）
    state_dim = 4
    embed_dim = 128
    num_layers = 8
    num_heads = 8
    ff_dim = 256
    vocab_size = 256

    # 1. 加载模型
    model = TransformerPi0FAST(
        state_dim=state_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        padding_id=pad_id
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. 构造输入
    if isinstance(state_vec, np.ndarray):
        state = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4)
    else:
        state = state_vec.float().to(device).unsqueeze(0)  # (1, 4)

    # 3. 生成token序列（自回归，遇到EOS或到达最大长度停止）
    tokens = [bos_id]
    for step in range(max_seq_len - 1):  # 已有BOS
        input_seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        attn_mask = (input_seq != pad_id).long()  # (1, T)
        with torch.no_grad():
            logits = model(state, input_seq, attn_mask)  # (1, T, vocab_size)
            next_token = logits[0, -1].argmax(-1).item()
        tokens.append(next_token)
        if next_token == eos_id:
            break
    return tokens

# ====== 用法示例 ======
if __name__ == "__main__":
    model_path = "train/trained_models/tinypi0fast_20250624_1734.pth"  # 替换为你的模型路径
    current_state = np.array([ 0.06656052, -0.01765201, -0.11830364,  0.219794  ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_tokens = infer_pi0fast_token_sequence(model_path, current_state, max_seq_len=25, device=device)
    print("Predicted token sequence:", pred_tokens)