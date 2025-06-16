import os

for idx in range(1001, 4001):
    os.system(f"python -m data_collection.main --idx {idx}")