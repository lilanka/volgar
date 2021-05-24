#pragma once 

#include <arrayfire.h>
#include "falcon/autograd/Tensor.h"

using namespace Falcon;
class F {
public:
  /* -------------------- ops ----------------------------*/
  Tensor add(const Tensor& a, const Tensor& b);
  Tensor sub(const Tensor& a, const Tensor& b);
  Tensor div(const Tensor& a, const float b);
  Tensor matmul(const Tensor& a, const Tensor& b); // Wx type; W-parameter
  Tensor mul0(const Tensor& a, const float b);
  Tensor mul1(const Tensor& a, const Tensor& b);
  Tensor pow(const Tensor& a, const float b);

  /* --------------------- activation functions ----------------*/
  Tensor relu(const Tensor& a);
  Tensor sigmoid(const Tensor& a);
};
