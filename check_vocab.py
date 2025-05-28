from processing_action_tokenizer import UniversalActionProcessor

# Load the trained processor and tokenizer
processor = UniversalActionProcessor.from_pretrained("tokenizer_trained/bpe_tokenizer")
tokenizer = processor.bpe_tokenizer

# Fetch vocab as dict: token → id
vocab = tokenizer.get_vocab()
inv_vocab = {v: k for k, v in vocab.items()}  # id → token

print(" Vocabulary Overview")

# Print tokens sorted by ID
for token_id in sorted(inv_vocab):
    token = inv_vocab[token_id]
    token_repr = repr(token)

    # Try mapping back to DCT space (if token is single character)
    if len(token) == 1:
        try:
            base = ord(token)
            dct_value = (base + processor.min_token) / processor.scale
            print(f"ID {token_id:<4} | Token {token_repr:<8} | DCT ≈ {dct_value:.3f}")
        except:
            print(f"ID {token_id:<4} | Token {token_repr:<8} | (non-decodable)")
    else:
        # Likely a BPE merge token
        merged_ids = [ord(c) for c in token]
        merged_values = [(i + processor.min_token) / processor.scale for i in merged_ids]
        values_str = ", ".join(f"{v:.2f}" for v in merged_values)
        print(f"ID {token_id:<4} | Token {token_repr:<8} | MERGE → [{values_str}]")

print("\n Total vocab size:", tokenizer.vocab_size)
