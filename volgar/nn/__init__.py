from volgar.tensor import Tensor
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
    pass

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.glorot_uniform(out_features, in_features)  
    self.bias = Tensor.zeros(out_features) if bias else None

  def __call__(self, x):
    out = dot(x, transpose(self.weight))
    return out 