"""
Pi0FlowMatchingTransformer: A Transformer-based model for mapping state sequences to continuous action sequences.

This model is designed for tasks such as reinforcement learning or control systems, where the input is a sequence of states
(e.g., from a CartPole environment) and the output is a sequence of continuous actions. The model uses a Transformer Encoder
to capture temporal dependencies in the state sequence and outputs a corresponding action sequence.

Author: Ruiming Wu
Date: 2025-05-08

Key Features:
- Input:
  - state_seq: (B, T, input_dim), where B is the batch size, T is the sequence length, and input_dim is the state dimension.
- Output:
  - action_out: (B, T, action_dim), where action_dim is the dimension of the continuous action.
- Position embeddings to encode temporal order.
- Transformer Encoder for capturing temporal dependencies.
"""

import torch
import torch.nn as nn

class Pi0FlowMatchingTransformer(nn.Module):
    def __init__(self,
                 input_dim=4,           # State dimension (e.g., CartPole = 4)
                 embed_dim=128,         # Embedding dimension
                 num_layers=4,          # Number of Transformer Encoder layers
                 num_heads=4,           # Number of attention heads
                 ff_dim=256,            # FeedForward network hidden dimension
                 action_dim=1,          # Output action dimension (continuous value)
                 max_seq_len=64):       # Maximum sequence length
        super().__init__()

        # State sequence embedding
        self.input_embedding = nn.Linear(input_dim, embed_dim)  # Map state input to embedding space
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)  # Positional embedding
        self.input_dropout = nn.Dropout(0.1)
        self.output_dropout = nn.Dropout(0.1)

        # Transformer Encoder (non-autoregressive structure)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            activation='gelu', batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer: Predict a continuous action (or velocity) for each time step
        self.output_layer = nn.Linear(embed_dim, action_dim)

        # Initialize weights
        self._init_weights()

    def forward(self, state_seq):
        """
        Forward pass of the model.

        Args:
            state_seq (torch.Tensor): Input state sequence of shape (B, T, input_dim).
        
        Returns:
            torch.Tensor: Output action sequence of shape (B, T, action_dim).
        """
        B, T, _ = state_seq.shape

        # State embedding + positional embedding
        x = self.input_embedding(state_seq)  # (B, T, embed_dim)
        pos_ids = torch.arange(T, device=state_seq.device).unsqueeze(0)  # (1, T)
        pos_embed = self.position_embedding(pos_ids)  # (1, T, embed_dim)
        x = x + pos_embed  # (B, T, embed_dim)

        # Transformer Encoder
        encoded = self.encoder(x)  # (B, T, embed_dim)

        # Output action sequence
        action_out = self.output_layer(encoded)  # (B, T, action_dim)
        return action_out

    def _init_weights(self):
        """
        Initialize weights using Xavier initialization.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
