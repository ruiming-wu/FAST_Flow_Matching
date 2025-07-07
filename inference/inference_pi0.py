import torch
import numpy as np
import matplotlib.pyplot as plt
from model.transformer_pi0 import TransformerPi0

def infer_pi0_action_sequence(model_path, state_vec, chunk_len=50, device='cpu', num_steps=10):
    """
    Inference for pi0 model: generate an action sequence using flow matching and Euler integration.
    Args:
        model_path: path to trained model weights
        state_vec: state vector (4,)
        chunk_len: length of action sequence to generate
        device: 'cpu' or 'cuda'
        num_steps: number of Euler steps for integration
    Returns:
        np.ndarray: generated action sequence, shape (chunk_len, 1)
    """
    # Hyperparameters (must match training)
    state_dim = 4
    action_dim = 1
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    ff_dim = 256

    # 1. Load model
    model = TransformerPi0(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        chunk_len=chunk_len
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Prepare input
    if isinstance(state_vec, np.ndarray):
        state = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4)
    else:
        state = state_vec.float().to(device).unsqueeze(0)  # (1, 4)

    # 3. Initialize action (Gaussian noise or zeros)
    noisy_action = torch.randn((1, chunk_len, action_dim), dtype=torch.float32, device=device)
    noisy_action = torch.clamp(noisy_action, -3.0, 3.0)
    delta = 1.0 / num_steps

    for step in range(num_steps):
        t = step * delta
        time_t = torch.full((1, chunk_len, 1), t, dtype=torch.float32, device=device)
        with torch.no_grad():
            flow = model(state, noisy_action, time_t)  # (1, chunk_len, action_dim)
        noisy_action = noisy_action + delta * flow  # Euler integration

    # Final action sequence
    action_seq = noisy_action.squeeze(0).cpu().numpy()  # (chunk_len, 1)
    return action_seq

# ====== Example usage ======
if __name__ == "__main__":
    model_path = "train/trained_models/tinypi0_034818062025.pth"  # Replace with your model path
    current_state = np.array([0.005, 0.0, -0.01, 0.0])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_actions = infer_pi0_action_sequence(model_path, current_state, chunk_len=50, device=device)
    print("Predicted action sequence:", pred_actions)