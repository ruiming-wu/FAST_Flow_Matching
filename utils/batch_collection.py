import os

for idx in range(1, 4001):
    os.system(f"python -m data_collection.main_pid --idx {idx}")