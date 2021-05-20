#include "falcon/autograd/Tensor.h"
#include "falcon/autograd/operators/operators.h"

#include <iostream>

namespace Falcon {

Tensor::Tensor(af::array data, bool requires_grad) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
}

Tensor::Tensor(af::array data, std::vector<Tensor> parents, bool requires_grad, int _op) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
  tensorData_->parents = std::move(parents); 
  tensorData_->_op = std::make_unique<int>(std::move(_op));
  if (requires_grad) {
    for (Tensor& parent: tensorData_->parents)
      tensorData_->grad.push_back(af::constant(0, parent.array().dims()));
  }
}

Tensor::Tensor(af::array data, std::vector<Tensor> parents, bool requires_grad, float _mul, int _op) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
  tensorData_->parents = std::move(parents); 
  tensorData_->_op = std::make_unique<int>(std::move(_op));
  tensorData_->_mul = std::make_unique<float>(std::move(_mul));
  if (requires_grad) {
    for (Tensor& parent: tensorData_->parents)
      tensorData_->grad.push_back(af::constant(0, parent.array().dims()));
  }
}
bool Tensor::isGradOn(const Tensor* other) const {
  if (tensorData_->requires_grad || (*other).tensorData_->requires_grad)  { return true; }
  return false; 
}

af::array& Tensor::array() const {
  return tensorData_->data;
}

/*
// not sure about this
void Tensor::grad() const {
  af::array total = af::constant(0, (tensorData_->grad[0]).dims());
  for (af::array& grad : tensorData_->grad) {
    total += grad;
  }
  af_print(total);
}
*/

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

void Tensor::gradFn() {
  switch (*tensorData_->_op) {
    case 0: std::cout << "addBackward()" << std::endl; break;
    case 1: std::cout << "subBackward()" << std::endl; break;
    case 2: std::cout << "divBackward()" << std::endl; break;
    case 3: std::cout << "mulBackward()" << std::endl; break;
    case 4: std::cout << "matmulBackward()" << std::endl; break;
  }
}

void Tensor::gradOp(const Tensor& tensor, const af::array& output_grad) {
  switch (*tensor.tensorData_->_op) {
    case 0: tensor.addBackward(output_grad); break;
    case 1: tensor.subBackward(output_grad); break;
    case 2: tensor.divBackward(output_grad); break;
    case 3: tensor.mulBackward(output_grad); break;
    case 4: tensor.matmulBackward(output_grad); break;
  }
}

void Tensor::addBackward(const af::array& output_grad) const {
  for (int i=0; i < tensorData_->parents.size(); i++) {
    tensorData_->grad[i] += output_grad;
    af_print(tensorData_->parents[i].array());
  }
}

void Tensor::subBackward(const af::array& output_grad) const{
  for (int i=0; i < tensorData_->parents.size(); i++) {
    tensorData_->grad[i] += output_grad;
  }
}

void Tensor::mulBackward(const af::array& output_grad) const {
  for (int i=0; i < tensorData_->parents.size(); i++) {
    tensorData_->grad[i] += output_grad * (*tensorData_->_mul);
  }
}

void Tensor::divBackward(const af::array& output_grad) const {
  for (int i=0; i < tensorData_->parents.size(); i++) {
    tensorData_->grad[i] += output_grad / (*tensorData_->_mul);
  }
}

void Tensor::matmulBackward(const af::array& output_grad) const {
  tensorData_->grad[1] += af::matmul(af::transpose(tensorData_->parents[0].array()), output_grad);
}

void Tensor::backward(const Tensor& tensor, const af::array& output_grad) {
  if (tensor.tensorData_->visited || 
      !(tensor.tensorData_->requires_grad) || 
      tensor.tensorData_->parents.size() == 0) { return; };

  tensor.tensorData_->visited = {true};
  gradOp(tensor, output_grad);
  for (int i=0; i< tensor.tensorData_->parents.size(); i++) {   
    backward(tensor.tensorData_->parents[i], tensor.tensorData_->grad[i]);
  }
}
void Tensor::backward(af::array initial_grad) {
  backward(*this, std::move(initial_grad));
}
}
