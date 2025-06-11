import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, random_split
import os
from datetime import datetime
import matplotlib.pyplot as plt

from models.transformer_pi0_fast import TransformerPi0FAST

# ========== Dataset Loader ==========

class Pi0FastDataset(Dataset):
    def __init__(self, state_path, token_path, mask_path):
        self.states = np.load(state_path)                    # (N, 4)
        self.tokens = np.load(token_path)                    # (N, T)
        self.attn_mask = np.load(mask_path)                  # (N, T)
        assert self.states.shape[0] == self.tokens.shape[0] == self.attn_mask.shape[0]
        self.N = self.states.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)         # (4,)
        tokens = torch.tensor(self.tokens[idx], dtype=torch.long)           # (T,)
        attn_mask = torch.tensor(self.attn_mask[idx], dtype=torch.long)     # (T,)
        target = tokens.clone()
        return state, tokens, attn_mask, target

# ========== Training Script ==========

def train_pi0_fast():
    # Set random seed
    seed = 27
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Hyperparameters
    state_dim = 4
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    ff_dim = 256
    vocab_size = 2048
    max_seq_len = 32
    num_epochs = 200
    batch_size = 32
    lr = 1.5e-4
    early_stop_patience = 20

    start_time = datetime.now()

    # Log file setup
    log_dir = os.path.join("training", "log")
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join("training", "trained_models")
    os.makedirs(model_dir, exist_ok=True)
    loss_pic_dir = os.path.join("training", "loss_pics")
    os.makedirs(loss_pic_dir, exist_ok=True)
    time_str = start_time.strftime("%H%M%d%m%Y")
    model_name = "transformerpi0fast"
    log_name = f"{model_name}_{time_str}"
    log_file_path = os.path.join(log_dir, f"{log_name}.txt")
    model_save_path = os.path.join(model_dir, f"{log_name}.pth")
    loss_pic_path = os.path.join(loss_pic_dir, f"{log_name}.png")
    log_file = open(log_file_path, "w", encoding="utf-8")

    def log_print(msg):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[Info] {now_str} {msg}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

    log_print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    log_print(f"Hyperparameters: state_dim={state_dim}, embed_dim={embed_dim}, num_layers={num_layers}, num_heads={num_heads}, ff_dim={ff_dim}, vocab_size={vocab_size}, max_seq_len={max_seq_len}, num_epochs={num_epochs}, batch_size={batch_size}, lr={lr}")

    # Dataset (合并加载后随机拆分)
    dataset = Pi0FastDataset(
        "data/states/all_states.npy",
        "data/tokens_pretrained/all_chunks_padded_tokens.npy",
        "data/tokens_pretrained/all_chunks_attention_mask.npy"
    )
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    log_print(f"Dataset size: total={total_size}, train={train_size}, val={val_size}, test={test_size}")

    # 随机拆分
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model
    model = TransformerPi0FAST(
        state_dim=state_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    log_print(f"Model initialized: {model.__class__.__name__}")

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.cuda.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        log_print(f"Starting epoch {epoch + 1}/{num_epochs} ...")
        for batch_idx, (states_batch, token_seqs_batch, attn_mask_batch, targets_batch) in enumerate(train_loader):
            states_batch = states_batch.to(device)
            token_seqs_batch = token_seqs_batch.to(device)
            attn_mask_batch = attn_mask_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(states_batch, token_seqs_batch, attn_mask_batch)  # (B, T, vocab_size)
                logits = logits.view(-1, logits.size(-1))       # (B*T, vocab_size)
                targets_batch = targets_batch.view(-1)          # (B*T)
                loss = criterion(logits, targets_batch)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                log_print(f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | Batch Loss: {loss.item():.6f}")

        avg_train_loss = total_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        log_print(f"Validating after epoch {epoch + 1} ...")
        with torch.no_grad():
            for val_batch_idx, (states_batch, token_seqs_batch, attn_mask_batch, targets_batch) in enumerate(val_loader):
                states_batch = states_batch.to(device)
                token_seqs_batch = token_seqs_batch.to(device)
                attn_mask_batch = attn_mask_batch.to(device)
                targets_batch = targets_batch.to(device)

                logits = model(states_batch, token_seqs_batch, attn_mask_batch)
                logits = logits.view(-1, logits.size(-1))
                targets_batch = targets_batch.view(-1)
                loss = criterion(logits, targets_batch)
                val_loss += loss.item()
                if (val_batch_idx + 1) % 20 == 0 or (val_batch_idx + 1) == len(val_loader):
                    log_print(f"Validation {val_batch_idx + 1}/{len(val_loader)} | Batch Loss: {loss.item():.6f}")
        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        log_print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            log_print(f"Best model saved at {model_save_path}!")

        # Early stopping
        if epoch - best_epoch >= early_stop_patience:
            log_print(f"No improvement for {early_stop_patience} epochs. Early stopping at epoch {epoch + 1}.")
            break

    # Final test evaluation
    log_print("Loading best model for final test evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for states_batch, token_seqs_batch, attn_mask_batch, targets_batch in test_loader:
            states_batch = states_batch.to(device)
            token_seqs_batch = token_seqs_batch.to(device)
            attn_mask_batch = attn_mask_batch.to(device)
            targets_batch = targets_batch.to(device)

            logits = model(states_batch, token_seqs_batch, attn_mask_batch)
            logits = logits.view(-1, logits.size(-1))
            targets_batch = targets_batch.view(-1)
            loss = criterion(logits, targets_batch)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    log_print(f"Final Test Loss: {avg_test_loss:.6f}")

    # 绘制loss曲线并保存
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.axhline(avg_test_loss, color='red', linestyle=':', label=f"Test Loss ({avg_test_loss:.4f})")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_pic_path)
    plt.close()
    log_print(f"Loss curve saved to {loss_pic_path}")

    end_time = datetime.now()
    total_time = end_time - start_time
    log_print(f"Training finished. Total time: {str(total_time)}")

    log_file.close()

if __name__ == "__main__":
    train_pi0_fast()