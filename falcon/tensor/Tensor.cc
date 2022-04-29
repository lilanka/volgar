#include "falcon/tensor/Tensor.h"
#include "falcon/autograd/Functional.h"

#include <iostream>

namespace Falcon {

F f;

Tensor::Tensor(af::array data, bool requires_grad) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
  if (requires_grad)
    tensorData_->grad = af::constant(0, tensorData_->data.dims());
}

Tensor::Tensor(af::array data, std::vector<Tensor> parents, bool requires_grad, int _op) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
  tensorData_->parents = std::move(parents); 
  tensorData_->_op = std::make_unique<int>(std::move(_op));
  if (requires_grad)
    tensorData_->grad = af::constant(0, tensorData_->data.dims());
}

Tensor::Tensor(af::array data, std::vector<Tensor> parents, bool requires_grad, float _mul, int _op) {
  tensorData_->data = std::move(data);
  tensorData_->requires_grad = requires_grad;
  tensorData_->parents = std::move(parents); 
  tensorData_->_op = std::make_unique<int>(std::move(_op));
  tensorData_->_mul = std::make_unique<float>(std::move(_mul));
  if (requires_grad)
    tensorData_->grad = af::constant(0, tensorData_->data.dims());
}
bool Tensor::isGradOn() const {
  if (tensorData_->requires_grad)
    return true;
  return false; 
}

af::array& Tensor::array() const {
  return tensorData_->data;
}

af::array Tensor::grad() const {
  if (!tensorData_->requires_grad)
    throw std::invalid_argument {"Gradient calculation deosn't allowed!"};
  return tensorData_->grad;
}

Tensor Tensor::operator+(const Tensor& tensor) {
  return f.add(*this, tensor);
}
Tensor Tensor::operator-(const Tensor& tensor) {
  return f.sub(*this, tensor);
}

Tensor Tensor::operator*(const float num) {
  return f.mul0(*this, num);
}

Tensor Tensor::operator*(const Tensor& tensor) {
  return f.mul1(*this, tensor);
}

Tensor Tensor::operator/(const float num) {
  return f.div(*this, num);
}

Tensor Tensor::operator^(const float num) {
  return f.pow(*this, num);
}

Tensor Tensor::matmul(const Tensor& tensor) {
  return f.matmul(*this, tensor);
}

Tensor Tensor::cat(const Tensor& tensor, int dim) {
  tensorData_->data = af::join(dim, tensorData_->data, tensor.array());
}

void Tensor::gradFn() {
  switch (*tensorData_->_op) {
    case 0: std::cout << "addBackward()" << std::endl; break;
    case 1: std::cout << "subBackward()" << std::endl; break;
    case 2: std::cout << "divBackward()" << std::endl; break;
    case 3: std::cout << "mulBackward0()" << std::endl; break;
    case 4: std::cout << "matmulBackward()" << std::endl; break;
    case 5: std::cout << "powBackward()" << std::endl; break;
    case 6: std::cout << "mulBackward1()" << std::endl; break;
    case 7: std::cout << "reluBackward()" << std::endl; break;
    case 8: std::cout << "sigmoidBackward()" << std::endl; break;
    case 9: std::cout << "tanhBackward()" << std::endl; break;
  }
}

void Tensor::gradOp(const Tensor& tensor, const af::array& output_grad) {
  switch (*tensor.tensorData_->_op) {
    case 0: tensor.addBackward(output_grad); break;
    case 1: tensor.subBackward(output_grad); break;
    case 2: tensor.divBackward(output_grad); break;
    case 3: tensor.mulBackward0(output_grad); break;
    case 4: tensor.matmulBackward(output_grad); break;
    case 5: tensor.powBackward(output_grad); break;
    case 6: tensor.mulBackward1(output_grad); break;
    case 7: tensor.reluBackward(output_grad); break;
    case 8: tensor.sigmoidBackward(output_grad); break;
    case 9: tensor.tanhBackward(output_grad); break;
  }
}

void Tensor::reluBackward(const af::array& output_grad) const {
  for (Tensor& parent: tensorData_->parents) {
    if (parent.tensorData_->requires_grad)
      parent.tensorData_->grad += output_grad * (parent.array() > parent.tensorData_->grad);
  }
}

void Tensor::sigmoidBackward(const af::array& output_grad) const {
  for (Tensor& parent: tensorData_->parents) {
    if (parent.tensorData_->requires_grad)
      parent.tensorData_->grad += output_grad * (array() * (af::constant(1, array().dims()) - array()));
  }
}

void Tensor::tanhBackward(const af::array& output_grad) const {
  for (Tensor& parent: tensorData_->parents) {
    if (parent.tensorData_->requires_grad)
      parent.tensorData_->grad += output_grad * (1 - array() * array());
  }
}
void Tensor::addBackward(const af::array& output_grad) const {
  for (Tensor& parent: tensorData_->parents) {
    if (parent.tensorData_->requires_grad)
      parent.tensorData_->grad += output_grad;
  }
}
void Tensor::mulBackward1(const af::array& output_grad) const {
  std::vector<Tensor> parents = tensorData_->parents;
  parents[0].tensorData_->grad += parents[1].array() * output_grad;
  parents[1].tensorData_->grad += parents[0].array() * output_grad;
}
void Tensor::subBackward(const af::array& output_grad) const{
  for (Tensor& parent: tensorData_->parents) {
    if (parent.tensorData_->requires_grad)
      parent.tensorData_->grad += output_grad;
  }
}

void Tensor::mulBackward0(const af::array& output_grad) const {
  std::vector<Tensor> parent = tensorData_->parents;
  parent[0].tensorData_->grad += output_grad * (*tensorData_->_mul);
}

void Tensor::powBackward(const af::array& output_grad) const {
  std::vector<Tensor> parent = tensorData_->parents;
  parent[0].tensorData_->grad += output_grad * (*tensorData_->_mul * (af::pow(parent[0].array(), *tensorData_->_mul - 1)));
}

void Tensor::divBackward(const af::array& output_grad) const {
  std::vector<Tensor> parent = tensorData_->parents;
  parent[0].tensorData_->grad += output_grad / (*tensorData_->_mul);
}

void Tensor::matmulBackward(const af::array& output_grad) const {
  std::vector<Tensor> parent = tensorData_->parents;
  parent[1].tensorData_->grad += af::matmulTN(parent[0].array(), output_grad);
}

void Tensor::backward(const Tensor& tensor, const af::array& output_grad) {
  if (tensor.tensorData_->visited || tensor.tensorData_->parents.size() == 0) {return;}
  tensor.tensorData_->visited = {true};
  gradOp(tensor, output_grad);
  for (Tensor& parent: tensor.tensorData_->parents) {  
    backward(parent, parent.tensorData_->grad);
  }
}
void Tensor::backward(af::array initial_grad) {
  tensorData_->grad = std::move(initial_grad);
  backward(*this, tensorData_->grad);
}

void Tensor::backward() {
  tensorData_->grad = af::constant(1, array().dims());
  backward(*this, tensorData_->grad);
}
} // namespace Falcon
