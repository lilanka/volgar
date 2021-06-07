#include "falcon/nn/Linear.h"
#include "falcon/tensor/Distributions.h"
#include "falcon/autograd/Functional.h"

#include <iostream>

namespace Falcon {

Linear::Linear(int in_, int out_, bool bias_) {
  layerData_->params = std::make_unique<Tensor>(normal(in_, out_, true));
  if (bias_)
    layerData_->bias = std::make_unique<Tensor>(normal(1, out_, true));
  else 
    layerData_->bias = std::make_unique<Tensor>(zeros(1, out_, false));
}

Tensor Linear::operator()(const Tensor& inputs) {
  F f;
  Tensor out = f.matmul(inputs, *layerData_->params) + *layerData_->bias;
  return out; 
}

Tensor Linear::weights() {
  return *layerData_->params;
}

Tensor Linear::bias() {
  return *layerData_->bias;
}
} // namespace Falcon
