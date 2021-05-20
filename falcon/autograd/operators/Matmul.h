#pragma once

#include <arrayfire.h>
#include "falcon/autograd/Tensor.h"

using namespace Falcon;
class Matmul {
public:
  Matmul() = default;

  /* 
   * Wx type   
    Args:
      W - a: parameter
      x - b: variable  
  */
  Tensor forward(const Tensor& a, const Tensor& b);
};
