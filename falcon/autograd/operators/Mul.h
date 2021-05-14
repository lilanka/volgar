#pragma once

#include <arrayfire.h>
#include "falcon/autograd/Tensor.h"

using namespace Falcon;
class Mul {
public:
  Mul() = default;
  Tensor forward(const Tensor& a, const float b);
};
