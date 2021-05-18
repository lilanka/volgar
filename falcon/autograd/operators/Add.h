#pragma once

#include <arrayfire.h>
#include "falcon/autograd/Tensor.h"

using namespace Falcon;
class Add {
public:
  Add() = default;
  Tensor forward(const Tensor& a, const Tensor& b);    
};
