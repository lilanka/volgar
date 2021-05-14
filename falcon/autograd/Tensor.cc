#include "falcon/autograd/Tensor.h"
#include "falcon/autograd/operators/operators.h"

#include <iostream>

namespace Falcon {

Sub sub;
Div div;
Mul mul;
Matmul _matmul;

Tensor::Tensor(af::array data, bool requires_grad) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
}
/*
Tensor::Tensor(af::array data, std::vector<Tensor> parents, bool requires_grad) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
  tensorData_->parents = std::move(parents); 
}
*/
bool Tensor::isGradOn(const Tensor* other) const {
  if (tensorData_->requires_grad || (*other).tensorData_->requires_grad)  { return true; }
  return false; 
}

af::array& Tensor::array() const {
  return tensorData_->data;
}

std::vector<Tensor> Tensor::parentSetUp(const Tensor* other) {
  std::vector<Tensor> parents {*this};
  if (!other) return parents;
  parents.insert(parents.end(), {*other});
  return parents;
}

Tensor Tensor::operator+(const Tensor& tensor) {
  Add add;
  Tensor results = add.forward(*this, tensor);
  results.tensorData_->parents = std::move(parentSetUp(&tensor));
  return results;
}
Tensor Tensor::operator-(const Tensor& tensor) {
  Sub sub;
  Tensor results = sub.forward(*this, tensor);
  results.tensorData_->parents = std::move(parentSetUp(&tensor));
  return results;
}

Tensor Tensor::operator*(const float num) {
  Mul mul;
  Tensor results = mul.forward(*this, num);
  results.tensorData_->parents = std::move(parentSetUp(nullptr));
  return results;
}

Tensor Tensor::operator/(const float num) {
  Div div;
  Tensor results = div.forward(*this, num);
  results.tensorData_->parents = std::move(parentSetUp(nullptr));
  return results;
}

Tensor Tensor::matmul(const Tensor& tensor) {
  Matmul _matmul;
  Tensor results = _matmul.forward(*this, tensor);
  results.tensorData_->parents = std::move(parentSetUp(&tensor));
  return results;
}

void Tensor::backward() {
  af::array initial_grad = af::constant(1, tensorData_->data.dims());
}
}
