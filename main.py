
from utils import L2_normalize_numpy
import numpy as np

x = np.random.randn(100, 128)
x_norm = L2_normalize_numpy(x)
