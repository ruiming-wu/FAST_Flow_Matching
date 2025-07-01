import torch
import torch.nn as nn

class TransformerPi0FAST(nn.Module):
    def __init__(self,
                 state_dim=4,           # state dimension
                 embed_dim=128,         # transformer embedding dimension
                 num_layers=4,          # number of transformer layers
                 num_heads=4,           # number of attention heads
                 ff_dim=256,            # feedforward hidden dimension
                 vocab_size=256,       # token vocabulary size
                 max_seq_len=25,        # max token sequence length
                 padding_id=2):         # padding token id
        super().__init__()

        # state embedding
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.state_dropout = nn.Dropout(0.1)

        # token embedding and position embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_id)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # decoder-only transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # output layer: token classification logits
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.output_dropout = nn.Dropout(0.1)

        # weight initialization
        self._init_weights()

    def forward(self, state, token_seq, attention_mask=None):
        """
        Args:
            state: (B, state_dim)
            token_seq: (B, T, 1) or (B, T), token id sequence
            attention_mask: (B, T), 1 for valid token, 0 for pad

        Returns:
            logits: (B, T, vocab_size)
        """
        # support (B, T, 1) input
        if token_seq.dim() == 3 and token_seq.shape[-1] == 1:
            token_seq = token_seq.squeeze(-1)  # (B, T)

        B, state_dim = state.shape
        B2, T = token_seq.shape
        assert B == B2, "Batch size mismatch between state and token_seq"

        # state embedding
        state_emb = self.state_embedding(state).unsqueeze(1)  # (B, 1, D)
        state_emb = self.state_dropout(state_emb)             # (B, 1, D)

        # token embedding + position embedding
        token_emb = self.token_embedding(token_seq)           # (B, T, D)
        pos_ids = torch.arange(T, device=token_seq.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos_ids)            # (1, T, D)
        decoder_input = token_emb + pos_emb                   # (B, T, D)

        # causal mask for autoregressive decoding
        causal_mask = torch.triu(torch.ones(T, T, device=token_seq.device), diagonal=1).bool()

        # padding mask
        if attention_mask is not None:
            padding_mask = attention_mask == 0  # (B, T), True for pad
        else:
            padding_mask = None

        # transformer decoder
        output = self.transformer(
            tgt=decoder_input,
            memory=state_emb,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask
        )  # (B, T, D)

        # output logits
        logits = self.output_layer(output)      # (B, T, vocab_size)
        logits = self.output_dropout(logits)
        return logits

    def _init_weights(self):
        """
        Xavier initialization for all weights
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)