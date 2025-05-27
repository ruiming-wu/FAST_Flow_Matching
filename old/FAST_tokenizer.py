import numpy as np
from transformers import AutoProcessor

tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
# Load your action data for tokenizer training
# Chunks do not need to be of the same length, we will use dummy data
action_data = np.load("new_data.npy")
action_data = np.random.rand(4000, 50, 14)

# Train the new tokenizer, depending on your dataset size this can take a few minutes
tokenizer = tokenizer.fit(action_data)