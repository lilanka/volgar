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

void Tensor::grad() const {
  af::array total = af::constant(0, (tensorData_->grad[0]).dims());
  for (af::array& grad : tensorData_->grad) {
    total += grad;
  }
  af_print(total);
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

void Tensor::backward(const Tensor& tensor, const af::array& output_grad) {
  if (tensor.tensorData_->visited) { return; };
  tensor.tensorData_->visited = {true};
  addBackward(output_grad); // for testing: add flexible method for all operations
  for (int i=0; i< tensor.tensorData_->parents.size(); i++) {   
    backward(tensor.tensorData_->parents[i], tensor.tensorData_->grad[i]);
  }
}

void Tensor::addBackward(const af::array& output_grad) {
  for (int i=0; i < tensorData_->parents.size(); i++) {
    tensorData_->grad[i] += output_grad;
  }
}

void Tensor::backward() {
  af::array initial_grad = af::constant(1, tensorData_->data.dims());
  backward(*this, initial_grad);
}
}
