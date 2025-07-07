import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import os
from datetime import datetime
import matplotlib.pyplot as plt

from model.transformer_pi0 import TransformerPi0

# ========== Flow Matching Loss ==========

def flow_matching_loss(model, state, noisy_action, time_t, gt_action):
    """
    Compute the flow matching loss for the model.
    """
    x_t = (1 - time_t) * noisy_action + time_t * gt_action  # (B, T, action_dim)
    target_flow = gt_action - noisy_action                  # (B, T, action_dim)
    pred_flow = model(state, x_t, time_t)                   # (B, T, action_dim)
    loss = nn.functional.mse_loss(pred_flow, target_flow)
    return loss

# ========== Dataset Loader ==========

class Pi0ChunkDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.state = data["state"].astype(np.float32)              # (N, 4)
        gt_action_flat = data["action_chunk"].astype(np.float32)   # (N, 50)
        self.gt_action = np.expand_dims(gt_action_flat, axis=-1)   # (N, 50, 1)

        self.noisy_action = np.clip(np.random.randn(*self.gt_action.shape).astype(np.float32), -3.0, 3.0)
        self.time_t = np.random.rand(*self.gt_action.shape).astype(np.float32)  # (N, 50, 1)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return (torch.tensor(self.state[idx]),               # (4,)
                torch.tensor(self.noisy_action[idx]),        # (50,1)
                torch.tensor(self.time_t[idx]),              # (50,1)
                torch.tensor(self.gt_action[idx]))           # (50,1)

# ========== Training Script ==========

def train_pi0():
    # Hyperparameters
    state_dim = 4
    action_dim = 1
    chunk_len = 50
    embed_dim = 128
    num_epochs = 200
    batch_size = 128
    lr = 2e-4

    start_time = datetime.now()

    # Directory setup
    log_dir = os.path.join("train", "logs")
    model_dir = os.path.join("train", "trained_models")
    loss_pic_dir = os.path.join("train", "loss_pics")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(loss_pic_dir, exist_ok=True)

    time_str = start_time.strftime("%Y%m%d_%H%M")
    model_name = "tinypi0"

    name = f"{model_name}_{time_str}"
    log_path = os.path.join(log_dir, f"{name}.txt")
    model_path = os.path.join(model_dir, f"{name}.pth")
    loss_pic_path = os.path.join(loss_pic_dir, f"{name}.png")

    log_file = open(log_path, "w", encoding="utf-8")  

    def log_print(msg):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[Info] {now_str} {msg}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

    log_print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    log_print(f"Hyperparameters: state_dim={state_dim}, action_dim={action_dim}, chunk_len={chunk_len}, embed_dim={embed_dim}, num_epochs={num_epochs}, batch_size={batch_size}, lr={lr}")

    # Model
    model = TransformerPi0(state_dim=state_dim,
                            action_dim=action_dim,
                            embed_dim=embed_dim,
                            chunk_len=chunk_len)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    log_print(f"Model initialized: {model.__class__.__name__}")

    # Dataset
    log_print("Loading dataset...")
    dataset = Pi0ChunkDataset('data/training_pairs_original.npz')
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    log_print(f"Dataset size: total={total_size}, train={train_size}, val={val_size}, test={test_size}")

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=6)

    train_loss_list = []
    val_loss_list = []

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_patience = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        log_print(f"Starting epoch {epoch + 1}/{num_epochs} ...")
        for batch_idx, batch in enumerate(train_loader):
            state, noisy_action, time_t, gt_action = [x.to(device) for x in batch]

            optimizer.zero_grad()
            loss = flow_matching_loss(model, state, noisy_action, time_t, gt_action)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        log_print(f"Validating after epoch {epoch + 1} ...")
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                state, noisy_action, time_t, gt_action = [x.to(device) for x in batch]

                loss = flow_matching_loss(model, state, noisy_action, time_t, gt_action)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        log_print(f"[Epoch {epoch+1}/{num_epochs}] Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f} | LR: {current_lr:.6g}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            log_print(f"Best model saved at {model_path}!")
        
        # Early stopping
        if epoch - best_epoch >= early_stop_patience:
            log_print(f"No improvement for {early_stop_patience} epochs. Early stopping at epoch {epoch + 1}.")
            break

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

    # Final test evaluation
    log_print("Loading best model for final test evaluation...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            state, noisy_action, time_t, gt_action = [x.to(device) for x in batch]

            loss = flow_matching_loss(model, state, noisy_action, time_t, gt_action)

            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    log_print(f"Final Test Loss: {avg_test_loss:.6f}")

    # ====== Plot and save loss curves ======
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.axhline(avg_test_loss, color='red', linestyle=':', label=f"Test Loss ({avg_test_loss:.4f})")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Flow Matching MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_pic_path)
    plt.close()

    end_time = datetime.now()
    total_time = end_time - start_time
    log_print(f"Training finished. Total time: {str(total_time)}")
    log_print(f"Best model saved to {model_path}")
    log_print(f"Training log saved to {log_path}")
    log_print(f"Loss curve saved to {loss_pic_path}")

    log_file.close()

if __name__ == "__main__":
    train_pi0()
