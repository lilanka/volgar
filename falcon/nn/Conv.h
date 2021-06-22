/* similar to the concepts presented in https://arxiv.org/pdf/2008.02454.pdf */

#pragma once 

#include <vector>
#include <string>
#include <cstdlib>
#include <memory>
#include <arrayfire.h>

#include "falcon/tensor/Tensor.h"

namespace Falcon {

class ConvND {
public:
  ConvND(int in_channels, 
         int out_channels, 
         std::vector<int> kernel_size, 
         std::vector<int> stride={}, 
         std::vector<int> padding={}
         bool bias=false)
  
  Tensor Conv2D::operator()(const Tensor& input);

private:
  int _C;
  Tensor kernals;
};
} // namespace falcon
