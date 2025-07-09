import time
import torch
import numpy as np
import os
from model.transformer_pi0 import TransformerPi0

def time_pi0(model_path, state_vec, chunk_len=50, device='cpu', num_steps=10):
    """
    Inference for pi0 model: generate an action sequence using flow matching and Euler integration.
    Returns:
        model_prepare_time, model_reconstruction_time, model_inference_time
    """
    state_dim = 4
    action_dim = 1
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    ff_dim = 256
    model_inference_time_list = []

    # 1. Load model
    model_prepare_start = time.time()
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
    model_prepare_end = time.time()
    model_prepare_time = model_prepare_end - model_prepare_start

    # 2. Prepare input
    if isinstance(state_vec, np.ndarray):
        state = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4)
    else:
        state = state_vec.float().to(device).unsqueeze(0)  # (1, 4)

    # 3. Initialize action (Gaussian noise or zeros)
    model_reconstruction_start = time.time()
    noisy_action = torch.randn((1, chunk_len, action_dim), dtype=torch.float32, device=device)
    noisy_action = torch.clamp(noisy_action, -3.0, 3.0)
    delta = 1.0 / num_steps

    for step in range(num_steps):
        t = step * delta
        time_t = torch.full((1, chunk_len, 1), t, dtype=torch.float32, device=device)
        with torch.no_grad():
            model_inference_start = time.time()
            flow = model(state, noisy_action, time_t)  # (1, chunk_len, action_dim)
            model_inference_end = time.time()
            model_inference_time_list.append(model_inference_end - model_inference_start)
        noisy_action = noisy_action + delta * flow  # Euler integration

    # Final action sequence
    action_seq = noisy_action.squeeze(0).cpu().numpy()  # (chunk_len, 1)
    model_reconstruction_end = time.time()
    model_reconstruction_time = model_reconstruction_end - model_reconstruction_start
    model_inference_time = np.mean(model_inference_time_list)
    return model_prepare_time, model_reconstruction_time, model_inference_time

# ====== Batch inference time statistics for all tinypi0 models ======
if __name__ == "__main__":
    model_dir = "train/trained_models"
    model_prefix = "tinypi0_"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_runs = 10000

    # Find all tinypi0 models
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith(model_prefix) and f.endswith(".pth")]
    model_files.sort()
    n_models = len(model_files)
    if n_models == 0:
        raise RuntimeError("No model found!")
    runs_per_model = total_runs // n_models

    print(f"Found {n_models} models, each will run {runs_per_model} times.")

    all_prepare_times = []
    all_reconstruction_times = []
    all_inference_times = []
    per_model_stats = []

    for model_idx, model_path in enumerate(model_files):
        prepare_time_list, reconstruction_time_list, inference_time_list = [], [], []
        for _ in range(runs_per_model):
            current_state = np.random.uniform(-0.5, 0.5, size=4)
            prepare_time, reconstruction_time, inference_time = time_pi0(
                model_path, current_state, device=device
            )
            prepare_time_list.append(prepare_time)
            reconstruction_time_list.append(reconstruction_time)
            inference_time_list.append(inference_time)
        all_prepare_times.extend(prepare_time_list)
        all_reconstruction_times.extend(reconstruction_time_list)
        all_inference_times.extend(inference_time_list)
        per_model_stats.append({
            "model": os.path.basename(model_path),
            "prepare_mean": np.mean(prepare_time_list) * 1000,
            "prepare_std": np.std(prepare_time_list) * 1000,
            "reconstruction_mean": np.mean(reconstruction_time_list) * 1000,
            "reconstruction_std": np.std(reconstruction_time_list) * 1000,
            "inference_mean": np.mean(inference_time_list) * 1000,
            "inference_std": np.std(inference_time_list) * 1000,
        })
        print(f"Model {model_idx + 1}/{n_models} finished.")

    print("==== Per-model statistics (ms) ====")
    for stat in per_model_stats:
        print(f"{stat['model']}: Prepare {stat['prepare_mean']:.2f}±{stat['prepare_std']:.2f}, "
              f"Reconstruction {stat['reconstruction_mean']:.2f}±{stat['reconstruction_std']:.2f}, "
              f"Inference {stat['inference_mean']:.2f}±{stat['inference_std']:.2f}")

    print("==== Overall statistics (all models combined, ms) ====")
    print("Average Prepare Time:", np.mean(all_prepare_times) * 1000, "±", np.std(all_prepare_times) * 1000, "ms")
    print("Average Reconstruction Time:", np.mean(all_reconstruction_times) * 1000, "±", np.std(all_reconstruction_times) * 1000, "ms")
    print("Average Inference Time:", np.mean(all_inference_times) * 1000, "±", np.std(all_inference_times) * 1000, "ms")