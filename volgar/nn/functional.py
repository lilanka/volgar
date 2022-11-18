import numpy as np

from ..tensor import Tensor
from ..common import *
from ..ops import OpType

def add(a, b):
  return a + b

def sub(a, b):
  return a - b

def mul(a, b):
  return a * b

def pow(a, b):
  return a ** b

def transpose(a):
  return Tensor(np.transpose(a.data), requires_grad=False)

def dot(a, b):
  # dim(a): nxm, dim(b): mxq
  if a.requires_grad or b.requires_grad:
    return Tensor(np.dot(a, b), requires_grad=True, parents=[a, b], op=OpType.DOT)
  return Tensor(np.dot(a.data, b.data), requires_grad=False)