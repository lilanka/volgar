from ..tensor import Tensor
from .functional import *

class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    self.kernel_size = (kernel_size, kernel_size)
    self.weight = Tensor.glorot_uniform(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
    self.bias = Tensor.zeros(out_channels) if bias else None
    self.kernels = Tensor.glorot_uniform(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])

  def __call__(self, x):
    # x should be in shape shape of [B, C, H, W]
    return self._conv2d(x)  

  def _conv2d(self, x):
    h = x[0][0][0]
    w = x[0][0][1]
    out = []
    for batch in x:
      channels = []
      for ch in batch:
        for h in range(x):
          pass