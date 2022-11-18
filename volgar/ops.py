from enum import Enum

class OpType(Enum):
  ADD = 1
  SUB = 2
  MUL = 3
  DIM = 4
  POW = 5
  DIV = 6
  DOT = 7 
  # activation functions
  SIGMOID = 8
  TANH = 9
  RELU = 10