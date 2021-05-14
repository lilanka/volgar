#pragma once

#include <arrayfire.h>
#include "falcon/autograd/Tensor.h"

using namespace Falcon;
class Div {
public:
  Div() = default;
  Tensor forward(const Tensor& a, const float b);
};
