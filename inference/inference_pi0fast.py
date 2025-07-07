import torch
import numpy as np
from model.transformer_pi0_fast import TransformerPi0FAST

def infer_pi0fast_token_sequence(model_path, state_vec, max_seq_len=25, device='cpu', bos_id=0, eos_id=1, pad_id=2):
    """
    Inference for pi0fast model: generate a token sequence (stop at EOS or max length).
    Args:
        model_path: path to trained model weights
        state_vec: state vector (4,)
        max_seq_len: maximum sequence length to generate
        device: 'cpu' or 'cuda'
        bos_id, eos_id, pad_id: special token ids
    Returns:
        List[int]: generated token sequence (including BOS and EOS)
    """
    # Hyperparameters (must match training)
    state_dim = 4
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    ff_dim = 256
    vocab_size = 256

    # 1. Load model
    model = TransformerPi0FAST(
        state_dim=state_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        padding_id=pad_id
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Prepare input
    if isinstance(state_vec, np.ndarray):
        state = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4)
    else:
        state = state_vec.float().to(device).unsqueeze(0)  # (1, 4)

    # 3. Autoregressive generation (stop at EOS or max length)
    tokens = [bos_id]
    for step in range(max_seq_len - 1):  # already have BOS
        input_seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        attn_mask = (input_seq != pad_id).long()  # (1, T)
        with torch.no_grad():
            logits = model(state, input_seq, attn_mask)  # (1, T, vocab_size)
            next_token = logits[0, -1].argmax(-1).item()
        tokens.append(next_token)
        if next_token == eos_id:
            break
    return tokens

# ====== Example usage ======
if __name__ == "__main__":
    model_path = "train/trained_models/tinypi0fast_20250624_1734.pth"  # Replace with your model path
    current_state = np.array([0.06656052, -0.01765201, -0.11830364, 0.219794])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_tokens = infer_pi0fast_token_sequence(model_path, current_state, max_seq_len=25, device=device)
    print("Predicted token sequence:", pred_tokens)