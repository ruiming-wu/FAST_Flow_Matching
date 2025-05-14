import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from models.transformer_pi0_fast import Pi0FASTTransformer

# Define the dataset class
class TokenSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, states, token_seqs, targets):
        self.states = states
        self.token_seqs = token_seqs
        self.targets = targets

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.token_seqs[idx], self.targets[idx]

# Initialize the model
model = Pi0FASTTransformer(input_dim=4, embed_dim=128, num_layers=4, num_heads=4, ff_dim=256, vocab_size=512, max_seq_len=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer, loss function, and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Example data (replace with actual data loading logic)
states = torch.randn(1000, 4)
token_seqs = torch.randint(0, 512, (1000, 64))
targets = torch.randint(0, 512, (1000, 64))

dataset = TokenSequenceDataset(states, token_seqs, targets)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Training loop with validation
num_epochs = 10
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for states_batch, token_seqs_batch, targets_batch in train_loader:
        states_batch = states_batch.to(device)
        token_seqs_batch = token_seqs_batch.to(device)
        targets_batch = targets_batch.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(states_batch, token_seqs_batch)
            logits = logits.view(-1, logits.size(-1))
            targets_batch = targets_batch.view(-1)
            loss = criterion(logits, targets_batch)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for states_batch, token_seqs_batch, targets_batch in val_loader:
            states_batch = states_batch.to(device)
            token_seqs_batch = token_seqs_batch.to(device)
            targets_batch = targets_batch.to(device)

            logits = model(states_batch, token_seqs_batch)
            logits = logits.view(-1, logits.size(-1))
            targets_batch = targets_batch.view(-1)
            loss = criterion(logits, targets_batch)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "pi0fast_transformer.pth")