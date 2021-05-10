#include "Tensor.h"
#include <iostream>

namespace Falcon {

Tensor::Tensor(af::array data, bool requires_grad) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
}

bool Tensor::isGradOn(bool val) {
  if (tensorData_->requires_grad || val) 
   return true;
  return false; 
}

af::array& Tensor::array() const {
  return tensorData_->data;
}

Tensor Tensor::operator+(const Tensor& tensor) {
  af::array temp_array = add.forward(array(), tensor.array());
  Tensor results = Tensor(temp_array, isGradOn(tensor.tensorData_->requires_grad)); 
  return results;
}

Tensor Tensor::operator-(const Tensor& tensor) {
  af::array temp_array = sub.forward(array(), tensor.array());
  Tensor results = Tensor(temp_array, isGradOn(tensor.tensorData_->requires_grad)); 
  return results;
}
Tensor Tensor::operator*(const float num) {
  af::array temp_array = mul.forward(array(), num);
  Tensor results = Tensor(temp_array, isGradOn(false)); 
  return results;
}

Tensor Tensor::operator/(const float num) {
  af::array temp_array = div.forward(array(), num);
  Tensor results = Tensor(temp_array, isGradOn(false)); 
  return results;
}
}
