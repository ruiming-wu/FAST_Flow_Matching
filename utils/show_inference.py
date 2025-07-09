import numpy as np
import torch
import matplotlib.pyplot as plt

from model.transformer_pi0_fast import TransformerPi0FAST
from fast.decoder import decoder
from scipy.fftpack import idct
from utils.build_corpus import GAMMA

from model.transformer_pi0 import TransformerPi0

def run_compare_inference_on_chunk(
    chunk_idx,
    pi0_model_path,
    pi0fast_model_path,
    tokenizer_path,
    original_npz_path="data/training_pairs_original.npz",
    max_seq_len=25,
    chunk_len=50,
    device="cpu"
):
    # 1. Load original data
    data = np.load(original_npz_path)
    states = data["state"]  # (N, 4)
    actions = data["action_chunk"]  # (N, 50)
    chunk_state = states[chunk_idx]
    chunk_action = actions[chunk_idx]

    # 2. pi0fast inference
    state_dim = 4
    vocab_size = 256
    embed_dim = 128
    pad_id = 2
    bos_id, eos_id = 0, 1
    # Load pi0fast model
    pi0fast_model = TransformerPi0FAST(
        state_dim=state_dim,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        padding_id=pad_id
    ).to(device)
    pi0fast_model.load_state_dict(torch.load(pi0fast_model_path, map_location=device, weights_only=True))
    pi0fast_model.eval()

    # Autoregressive token generation
    tokens = [bos_id]
    state_tensor = torch.tensor(chunk_state, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(max_seq_len - 1):
        input_seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = (input_seq != pad_id).long()
        with torch.no_grad():
            logits = pi0fast_model(state_tensor, input_seq, attn_mask)
            next_token = logits[0, -1].argmax(-1).item()
        tokens.append(next_token)
        if next_token == eos_id:
            break

    # Decode to int sequence
    token_ints = decoder(tokens, tokenizer_path)
    # Pad to chunk_len if needed
    if len(token_ints) < chunk_len:
        token_ints = token_ints + [0] * (chunk_len - len(token_ints))
    token_ints = token_ints[:chunk_len]

    # Dequantize and inverse DCT
    quantized = np.array(token_ints) / GAMMA
    pred_action_fast = idct(quantized, norm='ortho')[:chunk_len]

    # 3. pi0 inference
    action_dim = 1
    pi0_model = TransformerPi0(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        num_layers=4,
        num_heads=4,
        ff_dim=256,
        chunk_len=chunk_len
    ).to(device)
    pi0_model.load_state_dict(torch.load(pi0_model_path, map_location=device, weights_only=True))
    pi0_model.eval()

    state_tensor_pi0 = torch.tensor(chunk_state, dtype=torch.float32, device=device).unsqueeze(0)
    noisy_action = torch.randn((1, chunk_len, action_dim), dtype=torch.float32, device=device)
    noisy_action = torch.clamp(noisy_action, -3.0, 3.0)
    num_steps = 10
    delta = 1.0 / num_steps
    for step in range(num_steps):
        t = step * delta
        time_t = torch.full((1, chunk_len, 1), t, dtype=torch.float32, device=device)
        with torch.no_grad():
            flow = pi0_model(state_tensor_pi0, noisy_action, time_t)
        noisy_action = noisy_action + delta * flow
    pred_action_pi0 = noisy_action.squeeze(0).cpu().numpy().flatten()[:chunk_len]

    # 4. Compute MAE and MSE for both models
    mae_pi0 = np.mean(np.abs(pred_action_pi0 - chunk_action))
    mse_pi0 = np.mean((pred_action_pi0 - chunk_action) ** 2)
    mae_fast = np.mean(np.abs(pred_action_fast - chunk_action))
    mse_fast = np.mean((pred_action_fast - chunk_action) ** 2)

    # 5. Visualization
    plt.figure(figsize=(12, 5))
    plt.plot(chunk_action, label="Original Action", color='#90ee90')
    plt.plot(pred_action_pi0, label="Predicted Action (pi0)", linestyle='-.', color='blue')
    plt.plot(pred_action_fast, label="Predicted Action (pi0fast)", linestyle='--', color='red')
    plt.title(f"Chunk {chunk_idx}: Original vs. pi0 vs. pi0fast Predicted Action")
    plt.xlabel("Action Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    ax = plt.gca()
    ax.text(0.85, 0.95, 
            f"pi0\nMAE: {mae_pi0:.4f}\nMSE: {mse_pi0:.4f}", 
            color='blue', fontsize=11, ha='right', va='top', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    ax.text(0.98, 0.95, 
            f"pi0fast\nMAE: {mae_fast:.4f}\nMSE: {mse_fast:.4f}", 
            color='red', fontsize=11, ha='right', va='top', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    plt.show()

    print(f"pi0 MAE: {mae_pi0:.6f}, MSE: {mse_pi0:.6f}")
    print(f"pi0fast MAE: {mae_fast:.6f}, MSE: {mse_fast:.6f}")

if __name__ == "__main__":
    # Please specify the chunk index and model paths
    chunk_idx = 327  # Specify chunk index
    pi0_model_path = "train/trained_models/tinypi0_20250620_0133.pth"
    pi0fast_model_path = "train/trained_models/tinypi0fast_20250626_1939.pth"
    tokenizer_path = "fast/tokenizer/fast_tokenizer.json"
    run_compare_inference_on_chunk(
        chunk_idx, pi0_model_path, pi0fast_model_path, tokenizer_path
    )