import os
import time

for idx in range(501, 1001):
    os.system(f"python -m data_collection.main --idx {idx}")
    time.sleep(3.0)