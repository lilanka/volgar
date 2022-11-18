import math

import numpy as np

from .common import *
from .ops import OpType 

class Tensor:
  def __init__(self, data, requires_grad=False, parents=None, mul=None, op=None):
    self.data = np.array(data)
    self.requires_grad = requires_grad
    self.grad = np.zeros_like(self.data)
    self.op = op
    self.parents = parents
    self.mul = mul
    self.visited = False

  def __repr__(self):
    return f"Tensor: {self.data}, requires_grad={self.requires_grad}"

  def size(self):
    return self.data.shape 

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    if other.size() == self.size():  
      return Tensor(self.data + other.data, self.requires_grad or other.requires_grad, parents=[self, other], op=OpType.ADD)
    error("Dimensions should be equal")

  def __sub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    if other.size() == self.size():  
      return Tensor(self.data - other.data, self.requires_grad or other.requires_grad, parents=[self, other], op=OpType.SUB)
    error("Dimensions should be equal")

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    """
    if self.size()[-1] == other.size()[0]:
      # elementwise multiplication
      print("passed")
      return Tensor(np.dot(self.data, other.data), self.requires_grad or other.requires_grad, parents=[self, other], op=OpType.DOT)
    """
    return Tensor(self.data * other.data, self.requires_grad or other.requires_grad, parents=[self], mul=other.data, op=OpType.MUL)

  def __pow__(self, other):
    # elementwise power 
    # other: can be a one dimensional tensor or a natural number 
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Tensor(self.data ** other.data, self.requires_grad or other.requires_grad, parents=[self], mul=other.data, op=OpType.POW)

  def backward(self):
    self.grad = np.ones_like(self.data)
    self._backward(self)

  def _backward(self, curr_tensor):
    if (curr_tensor.visited or curr_tensor.parents == None): return
    curr_tensor.visited = True
    #self._debug_grad_fn(curr_tensor)
    # taking bacward gradients
    self._grad_optype(curr_tensor)
    
    for parent in curr_tensor.parents:
      if parent.requires_grad: 
        self._backward(parent)

  def _grad_optype(self, tensor):
    match tensor.op:
      case OpType.ADD: self._add_backward(tensor)
      case OpType.SUB: self._sub_backward(tensor)
      case OpType.MUL: self._mul_backward(tensor)
      case OpType.DOT: self._dot_backward(tensor)
      case OpType.DIV: self._div_backward(tensor)
      case OpType.POW: self._pow_backward(tensor)
      case OpType.RELU: self._relu_backward(tensor)

  def _debug_grad_fn(self, tensor):
    # for debuging purposes
    match tensor.op:
      case OpType.ADD: print("add_backward") 
      case OpType.SUB: print("sub_backward") 
      case OpType.MUL: print("mul_backward") 
      case OpType.DIV: print("div_backward") 
      case OpType.POW: print("pow_backward") 

  # -- backward functions -- #
  # todo: add backward functions for activation functions

  def _add_backward(self, tensorj):
    for parent in tensor.parents:
      if parent.requires_grad: parent.grad += tensor.grad

  def _sub_backward(self, tensor):
    if tensor.parents[0].requires_grad: tensor.parents[0].grad += tensor.grad
    if tensor.parents[1].requires_grad: tensor.parents[1].grad -= tensor.grad

  def _mul_backward(self, tensor):
    if tensor.mul is not None:
      if tensor.parents[0].requires_grad: tensor.parents[0].grad += tensor.grad * tensor.mul
    """
    else:
      if self.parents[0].requires_grad: self.parents[0].grad += self.parents[1].data * grad
      if self.parents[1].requires_grad: self.parents[1].grad += self.parents[0].data * grad
    """

  # https://en.wikipedia.org/wiki/Vector_calculus_identities#cite_note-4:~:text=.-,Dot%20product%20rule,-%5Bedit%5D
  def _dot_backward(self, other):
    pass
    """
    if tensor.parents[0].requires_grad and tensor.parents[1].requires_grad:
      tensor.parents
    """

  def _div_backward(self, grad):
    if tensor.mul is not None:
      if tensor.parents[0].requires_grad: tensor.parents[0].grad += tensor.grad / tensor.mul
    """
    else:
      if self.parents[0].requires_grad: self.parents[0].grad += (1. / self.parents[1].data) * grad
      if self.parents[1].requires_grad: self.parents[1].grad += (self.parents[0].data / self.parents[1].data ** 2) * grad
    """

  def _pow_backward(self, tensor):
    # for operations where tensor to the power scalar
    if tensor.mul is not None:
      tensor.parents[0].grad += tensor.grad * tensor.mul * (tensor.parents[0].data ** (tensor.mul - 1))

  def _relu_backward(self, tensor):
    if tensor.parents[0].requires_grad:
      tensor.parents[0].grad += 0 if (tensor.data <= 0).all() == True else tensor.grad

  @classmethod
  def zeros(cls, *shape, **kwargs):
    return cls(np.zeros(shape, dtype=np.float32), **kwargs)

  # todo: look into kaiming initialization methods
  @classmethod
  def glorot_uniform(cls, *shape, **kwargs):
    return cls((np.random.default_rng().random(size=shape, dtype=np.float32) * 2 - 1) * ((6 / (shape[0] + math.prod(shape[1:]))) ** 0.5), **kwargs) 