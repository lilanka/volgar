from ..tensor import Tensor
from .functional import *

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.glorot_uniform(out_features, in_features)  
    self.bias = Tensor.zeros(out_features) if bias else None

  def __call__(self, x):
    return dot(x, transpose(self.weight))