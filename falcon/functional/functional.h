#pragma once

#include "falcon/common.h"
#include "falcon/tensor/tensor.h"

namespace falcon {

// Functional operations 
class F {
public:
  tensor add(const tensor& x, const tensor& y);
  tensor sub(const tensor& x, const tensor& y);
  tensor div(const tensor& x, const float number);
  tensor div(const tensor& x, const tensor& y);
  tensor mul(const tensor& x, const float number);
  tensor mul(const tensor& x, const tensor& y);
  tensor pow(const tensor& x, const float number);

  // Sin functions
  tensor sin(const tensor& x);
  tensor cos(const tensor& x);
};

} // namespace falcon