"""
Pi0FASTTransformer: A Transformer-based model for autoregressive token sequence generation.

This model is designed for tasks such as sequence generation or token prediction, where the input consists of a state vector
and a token sequence, and the output is the logits for the next token in the sequence. The model uses a Transformer Decoder
to capture dependencies in the token sequence while conditioning on the input state.

Author: Ruiming Wu
Date: 2025-05-08

Key Features:
- Input:
  - state: (B, input_dim), where B is the batch size and input_dim is the state dimension.
  - token_seq: (B, T), where T is the sequence length.
- Output:
  - logits: (B, T, vocab_size), where vocab_size is the size of the token vocabulary.
- Position embeddings to encode temporal order.
- Transformer Decoder for autoregressive sequence modeling.
"""

import torch
import torch.nn as nn

class Pi0FASTTransformer(nn.Module):
    def __init__(self, 
                 input_dim=4,              # State dimension (e.g., CartPole = 4)
                 embed_dim=128,            # Embedding dimension
                 num_layers=4,             # Number of Transformer Decoder layers
                 num_heads=4,              # Number of attention heads
                 ff_dim=256,               # FeedForward network hidden dimension
                 vocab_size=512,           # Token vocabulary size (e.g., from BPE encoding)
                 max_seq_len=64):          # Maximum token sequence length
        super().__init__()

        # State embedding (used as memory input for the Transformer Decoder)
        self.state_embedding = nn.Linear(input_dim, embed_dim)
        self.state_dropout = nn.Dropout(0.1)

        # Token embedding + position embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Decoder-only Transformer (GPT-like structure)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            batch_first=True, activation="gelu", dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer: logits for token classification
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.output_dropout = nn.Dropout(0.1)

        # Initialize weights
        self._init_weights()

    def forward(self, state, token_seq):
        """
        Forward pass of the model.

        Args:
            state (torch.Tensor): Input state vector of shape (B, input_dim).
            token_seq (torch.Tensor): Input token sequence of shape (B, T).

        Returns:
            torch.Tensor: Logits for token classification of shape (B, T, vocab_size).
        """
        B, T = token_seq.shape

        # (B, 1, embed_dim), used as memory input for the Transformer Decoder
        state_embed = self.state_embedding(state).unsqueeze(1)
        state_embed = self.state_dropout(state_embed)

        # Token embedding + position embedding
        tok_embed = self.token_embedding(token_seq)                       # (B, T, embed_dim)
        pos_ids = torch.arange(T, device=token_seq.device).unsqueeze(0)  # (1, T)
        pos_embed = self.position_embedding(pos_ids)                     # (1, T, embed_dim)
        decoder_input = tok_embed + pos_embed                            # (B, T, embed_dim)

        # Causal mask: lower triangular mask (ensures autoregressive behavior)
        causal_mask = torch.triu(torch.ones(T, T, device=token_seq.device), diagonal=1).bool()
        assert causal_mask.shape == (T, T), "Causal mask shape mismatch"

        # Transformer Decoder
        transformer_out = self.transformer(tgt=decoder_input,
                                           memory=state_embed,
                                           tgt_mask=causal_mask)

        # Output logits (for CrossEntropyLoss)
        logits = self.output_layer(transformer_out)  # (B, T, vocab_size)
        logits = self.output_dropout(logits)
        return logits

    def _init_weights(self):
        """
        Initialize weights using Xavier initialization.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)