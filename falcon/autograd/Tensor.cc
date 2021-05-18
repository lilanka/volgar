#include "falcon/autograd/Tensor.h"
#include "falcon/autograd/operators/operators.h"

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
  if (requires_grad) {
    for (Tensor& _: tensorData_->parents)
      tensorData_->grad.push_back(af::constant(0, tensorData_->data.dims()));
  }
}

bool Tensor::isGradOn(const Tensor* other) const {
  if (tensorData_->requires_grad || (*other).tensorData_->requires_grad)  { return true; }
  return false; 
}

af::array& Tensor::array() const {
  return tensorData_->data;
}

Tensor Tensor::operator+(const Tensor& tensor) {
  Add add;
  return add.forward(*this, tensor);
}
Tensor Tensor::operator-(const Tensor& tensor) {
  Sub sub;
  return sub.forward(*this, tensor);
}

Tensor Tensor::operator*(const float num) {
  Mul mul;
  return mul.forward(*this, num);
}

Tensor Tensor::operator/(const float num) {
  Div div;
  return div.forward(*this, num);
}

Tensor Tensor::matmul(const Tensor& tensor) {
  Matmul _matmul;
  return _matmul.forward(*this, tensor);
}

void Tensor::backward() {
  af::array initial_grad = af::constant(1, tensorData_->data.dims());
}
}
