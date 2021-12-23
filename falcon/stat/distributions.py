import math
import numpy as np

from utils import Utils 

class Distributions(object):
  
  @classmethod
  def binomial(self, n: int, y: int, data: list) -> list:
    return self._ncr(n, y) * (data**y) * (1 - data)**(n-y) 

  def _ncr(n: int, r: int) -> float:
    return math.factorial(n)/(math.factorial(n-r) * math.factorial(r))       
