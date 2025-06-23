import torch
import numpy as np
import matplotlib.pyplot as plt
from model.transformer_pi0 import TransformerPi0

def infer_pi0_action_sequence(model_path, state_vec, chunk_len=50, device='cpu', num_steps=10):
    # 超参数（需与训练一致）
    state_dim = 4
    action_dim = 1
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    ff_dim = 256

    # 1. 加载模型
    model = TransformerPi0(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        chunk_len=chunk_len
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()

    # 2. 构造输入
    if isinstance(state_vec, np.ndarray):
        state = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4)
    else:
        state = state_vec.float().to(device).unsqueeze(0)  # (1, 4)

    # 3. 初始化动作（论文用高斯噪声，或全0也可）
    noisy_action = torch.randn((1, chunk_len, action_dim), dtype=torch.float32, device=device)
    noisy_action = torch.clamp(noisy_action, -3.0, 3.0)
    delta = 1.0 / num_steps

    for step in range(num_steps):
        t = step * delta
        time_t = torch.full((1, chunk_len, 1), t, dtype=torch.float32, device=device)
        with torch.no_grad():
            flow = model(state, noisy_action, time_t)  # (1, chunk_len, action_dim)
        noisy_action = noisy_action + delta * flow  # 欧拉积分

    # 最终动作
    action_seq = noisy_action.squeeze(0).cpu().numpy()  # (chunk_len, 1)
    return action_seq

# ====== 用法示例 ======
if __name__ == "__main__":
    model_path = "train/trained_models/tinypi0_034818062025.pth"
    current_state = np.array([0.005, 0.0, -0.01, 0.0])
    pred_actions = infer_pi0_action_sequence(model_path, current_state, chunk_len=50, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("Predicted action sequence shape:", pred_actions.shape)
    print("First 5 actions:\n", pred_actions[:5])

    # 绘制动作序列
    plt.figure(figsize=(8, 4))
    plt.plot(pred_actions, marker='o')
    plt.title("Predicted Action Sequence")
    plt.xlabel("Step")
    plt.ylabel("Action Value")
    plt.grid(True)
    plt.show()