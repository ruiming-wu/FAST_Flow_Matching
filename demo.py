import numpy as np
import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is NOT available. Using CPU.")
    try:
        import torch.backends.cudnn as cudnn
        print("cudnn enabled:", cudnn.enabled)
    except Exception as e:
        print("cudnn check failed:", e)
    # 检查常见原因
    import sys
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    try:
        import numpy
        print("Numpy version:", numpy.__version__)
    except ImportError:
        print("Numpy not installed.")
    print("Possible reasons: No compatible GPU, CUDA driver not installed, or PyTorch not built with CUDA.")