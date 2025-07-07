# 🧩 model Module

This directory contains the neural network architectures used for both the tiny pi0 and the tiny pi0fast.

---

## Overview

The `model` module provides PyTorch implementations of transformer-based sequence models for action prediction and token sequence modeling in the Inverted Pendulum control task.

---

## Main Files & Their Functions

- **transformer_pi0.py**  
  - Purpose: Implements the transformer model for the tiny pi0 controller, which predicts a sequence of continuous actions given the current state.
  - Key Features:
    - Uses state, action, and time embeddings.
    - Decoder-only transformer (self-attention) for action sequence modeling.
    - Outputs a flow vector for each action step.
    - Designed for continuous action prediction (e.g., [N, 50, 1]).

- **transformer_pi0_fast.py**  
  - Purpose: Implements the transformer model for the tiny pi0fast controller, which predicts a sequence of discrete tokens representing compressed/quantized actions.
  - Key Features:
    - Uses state, token, and position embeddings.
    - Decoder-only transformer for autoregressive token sequence modeling.
    - Outputs logits over the token vocabulary for each step.
    - Designed for efficient modeling of quantized action sequences (e.g., [N, 25] tokens).

---

## Typical Usage

Both models are designed to be imported and used in training or inference scripts. Example usage:

```python
from model.transformer_pi0 import TransformerPi0
from model.transformer_pi0_fast import TransformerPi0FAST

# For pi0 (continuous actions)
model_pi0 = TransformerPi0(state_dim=4, action_dim=1, chunk_len=50)

# For pi0-FAST (tokenized actions)
model_pi0fast = TransformerPi0FAST(state_dim=4, vocab_size=256, max_seq_len=25)