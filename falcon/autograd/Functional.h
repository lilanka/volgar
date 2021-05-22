#pragma once 

#include <arrayfire.h>
#include "falcon/autograd/Tensor.h"

using namespace Falcon;
class F {
public:
  Tensor add(const Tensor& a, const Tensor& b);
  Tensor sub(const Tensor& a, const Tensor& b);
  Tensor div(const Tensor& a, const float b);
  Tensor matmul(const Tensor& a, const Tensor& b); // Wx type; W-parameter
  Tensor mul0(const Tensor& a, const float b);
  Tensor mul1(const Tensor& a, const Tensor& b);

  Tensor relu(const Tensor& a);
};
