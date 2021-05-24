#pragma once

#include <arrayfire.h>
#include "falcon/tensor/Tensor.h"

namespace Falcon {

// make Tensor <- normal distribution
Tensor normal( int d1=1, int d2=1, int d3=1, int d4=1, bool requires_grad=false) {
  return Tensor(af::randn(d1, d2, d3, d4), requires_grad);
}

// make Tensor <- zeros inside
Tensor zeros(int d1=1, int d2=1, int d3=1, int d4=1, bool requires_grad=false) {
  return Tensor(af::constant(0, d1, d2, d3, d4), requires_grad);
}
} // namespace Falcon
