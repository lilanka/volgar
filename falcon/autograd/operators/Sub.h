#pragma once

#include <arrayfire.h>
#include "falcon/autograd/Tensor.h"

using namespace Falcon;
class Sub {
public:
  Sub() = default;
  Tensor forward(const Tensor& a, const Tensor& b);
};
