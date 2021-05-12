#include "Tensor.h"
#include <iostream>

namespace Falcon {

Tensor::Tensor(af::array data, bool requires_grad) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
}

Tensor::Tensor(af::array data, std::vector<Tensor> parents, bool requires_grad) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
  tensorData_->parents = std::move(parents); 
}

bool Tensor::isGradOn(bool val) {
  if (tensorData_->requires_grad || val) 
    return true;
  return false; 
}

af::array& Tensor::array() const {
  return tensorData_->data;
}

std::vector<Tensor> Tensor::parentSetUp(const Tensor* other) {
  std::vector<Tensor> parents;
  if (!other) 
    return parents;
  parents.insert(parents.end(), {*this, *other});
  return parents;
}

Tensor Tensor::operator+(const Tensor& tensor) {
  Tensor results = Tensor(add.forward(array(), tensor.array()), \
      parentSetUp(&tensor), isGradOn(tensor.tensorData_->requires_grad));
  return results;
}

Tensor Tensor::operator-(const Tensor& tensor) {
  Tensor results = Tensor(sub.forward(array(), tensor.array()), \
      parentSetUp(&tensor), isGradOn(tensor.tensorData_->requires_grad)); 
  return results;
}
Tensor Tensor::operator*(const float num) {
  Tensor results = Tensor(mul.forward(array(), num), \
      parentSetUp(nullptr), isGradOn(false)); 
  return results;
}

Tensor Tensor::operator/(const float num) {
  Tensor results = Tensor(div.forward(array(), num), \
      parentSetUp(nullptr), isGradOn(false)); 
  return results;
}

Tensor Tensor::matmul(const Tensor& tensor) {
  Tensor results = Tensor(_matmul.forward(array(), tensor.array()), \
      parentSetUp(&tensor), isGradOn(tensor.tensorData_->requires_grad)); 
  return results;
}
}
