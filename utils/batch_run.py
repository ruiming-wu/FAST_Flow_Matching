import os
import time

for idx in range(60, 101):
    os.system(f"python -m data_collection.main --idx {idx}")
    time.sleep(5.0)