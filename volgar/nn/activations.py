import numpy as np

from ..tensor import Tensor
from ..common import *
from ..ops import OpType

class Sigmoid:
  def __call__(self, x):
    return Tensor(1 / (1 + np.exp(-x.data)), x.requires_grad, parents=[x], op=OpType.SIGMOID) 

class Tanh:
  def __call__(self, x):
    return Tensor(2 * (1 / (1 + np.exp((-2) * x.data))) - 1, x.requires_grad, parents=[x], op=OpType.TANH)

class ReLU:
  def __call__(self, x):
    return Tensor(np.maximum(0, x.data), x.requires_grad, parents=[x], op=OpType.RELU)