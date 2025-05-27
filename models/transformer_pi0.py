import torch
import torch.nn as nn

class TransformerPi0(nn.Module):
    def __init__(self,
                 state_dim=4,          # e.g., CartPole state dim
                 action_dim=1,         # e.g., CartPole action dim
                 embed_dim=128,        # transformer embedding dim
                 num_layers=4,         # transformer layers
                 num_heads=4,          # attention heads
                 ff_dim=256,           # feedforward hidden dim
                 chunk_len=50):        # action chunk length
        super().__init__()

        # Embeddings
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.action_embedding = nn.Linear(action_dim, embed_dim)
        self.time_embedding = nn.Linear(1, embed_dim)  # scalar time → embed

        # Positional embedding
        self.position_embedding = nn.Embedding(chunk_len + 1, embed_dim)  # +1 for state token

        # Decoder-only transformer (self-attention only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: flow vector
        self.flow_output = nn.Linear(embed_dim, action_dim)

    def forward(self, state, noisy_action, time_t):
        """
        Args:
            state: (B, state_dim)
            noisy_action: (B, T, action_dim)
            time_t: (B, T, 1) normalized [0,1]

        Returns:
            flow: (B, T, action_dim)
        """
        B, T, _ = noisy_action.shape

        # State embedding → expand across sequence
        state_emb = self.state_embedding(state).unsqueeze(1)  # (B, 1, D)

        # Action + time embedding
        action_emb = self.action_embedding(noisy_action)  # (B, T, D)
        time_emb = self.time_embedding(time_t)           # (B, T, D)

        # Positional embedding
        pos_ids = torch.arange(T + 1, device=state.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)       # (1, T+1, D)

        # Combine: [state] + [action + time]
        combined = torch.cat([state_emb, action_emb + time_emb], dim=1)  # (B, T+1, D)
        combined = combined + pos_emb  # add position info

        # Causal mask
        causal_mask = torch.triu(torch.ones(T + 1, T + 1, device=state.device), diagonal=1).bool()

        # Decoder-only Transformer
        output = self.transformer(combined, mask=causal_mask)

        # Predict flow (skip state token, only action part)
        flow = self.flow_output(output[:, 1:, :])  # (B, T, action_dim)

        return flow

