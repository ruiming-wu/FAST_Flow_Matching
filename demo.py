import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


x = np.random.randn(8)
y1 = dct(x)               # 默认 norm=None
y2 = dct(x, norm='ortho') # 正交归一化

# 逆变换
x1 = idct(y1)             # 不能还原原信号
x2 = idct(y2, norm='ortho') # 可以精确还原原信号

print("Original signal:", x)
print("DCT (default norm):", y1)
print("DCT (ortho norm):", y2)
print("IDCT (default norm):", x1)
print("IDCT (ortho norm):", x2)