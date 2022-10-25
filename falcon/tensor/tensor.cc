#include <iostream>
#include <boost/variant.hpp>

#include "tensor.h"
#include "falcon/functional/functional.h"

namespace falcon {

tensor::tensor(af::array&& data, const bool requires_grad) {
  tensor_data->data = data;      
  tensor_data->requires_grad = requires_grad;
  if (requires_grad)
    tensor_data->grad = af::constant(0, tensor_data->data.dims());
}

tensor::tensor(af::array&& data, const bool requires_grad, std::vector<tensor> parents, OpType op) {
  tensor_data->data = data;
  tensor_data->requires_grad = requires_grad;
  tensor_data->parents = parents;
  tensor_data->op = op;
  if (requires_grad)
    tensor_data->grad = af::constant(0, tensor_data->data.dims());
}

tensor::tensor(af::array&& data, const bool requires_grad, std::vector<tensor> parents, float mul, OpType op) {
  tensor_data->data = data;
  tensor_data->requires_grad = requires_grad;
  tensor_data->parents = parents;
  tensor_data->op = op;
  tensor_data->mul = mul;
  if (requires_grad)
    tensor_data->grad = af::constant(0, tensor_data->data.dims());
}

af::array& tensor::data() const {
  return tensor_data->data; 
}

af::array& tensor::grad() const {
  return tensor_data->grad; 
}

bool tensor::is_grad_on() const {
  if (tensor_data->requires_grad) 
    return true;
  return false;
}

tensor tensor::operator+(const tensor& other) {
  F f;
  return f.add(*this, other);
}

tensor tensor::operator-(const tensor& other) {
  F f;
  return f.sub(*this, other);
}

tensor tensor::operator/(const float number) {
  F f;
  return f.div(*this, number);
}

tensor tensor::operator*(const float number) {
  F f;
  return f.mul(*this, number);
}

tensor tensor::operator^(const float number) {
  F f;
  return f.pow(*this, number);
}

void tensor::add_backword(const af::array& output_grad) const {
  for (tensor& parent: tensor_data->parents) {
    if (parent.tensor_data->requires_grad)
      parent.tensor_data->grad += output_grad;
  }
}

void tensor::sub_backword(const af::array& output_grad) const {
  std::vector<tensor> parents = tensor_data->parents;
  tensor_data->parents[0].tensor_data->grad += output_grad;
  tensor_data->parents[1].tensor_data->grad -= output_grad;
}

void tensor::mul_backword(const af::array& output_grad) const {
  std::vector<tensor> parents = tensor_data->parents;
  if (parents.size() == 1)
    tensor_data->parents[0].tensor_data->grad += output_grad * tensor_data->mul;  
  else {
    tensor_data->parents[0].tensor_data->grad += parents[1].data() * output_grad;
    tensor_data->parents[1].tensor_data->grad += parents[0].data() * output_grad;
  }
}

void tensor::div_backword(const af::array& output_grad) const {
  tensor_data->parents[0].tensor_data->grad += output_grad / tensor_data->mul;
}

void tensor::pow_backword(const af::array& output_grad) const {
  std::vector<tensor> parents = tensor_data->parents;
  tensor_data->parents[0].tensor_data->grad += output_grad * tensor_data->mul * \
    af::pow(parents[0].data(), tensor_data->mul - 1);
}

void tensor::backword(const tensor& curr_tensor, const af::array& output_grad) {
#define GRAD_OpType() {                                               \
  switch (curr_tensor.tensor_data->op) {                              \
    case OpType::ADD: curr_tensor.add_backword(output_grad); break;   \
    case OpType::SUB: curr_tensor.sub_backword(output_grad); break;   \
    case OpType::MUL: curr_tensor.mul_backword(output_grad); break;   \
    case OpType::DIV: curr_tensor.div_backword(output_grad); break;   \
    case OpType::POW: curr_tensor.pow_backword(output_grad); break;   \
  }                                                                   \
}
#define GRAD_FN() {                                                       \
  switch (curr_tensor.tensor_data->op) {                                  \
    case OpType::ADD: std::cout << "add_backword()" << std::endl; break;  \
    case OpType::SUB: std::cout << "sub_backword()" << std::endl; break;  \
    case OpType::MUL: std::cout << "mul_backword()" << std::endl; break;  \
    case OpType::DIV: std::cout << "div_backword()" << std::endl; break;  \
    case OpType::POW: std::cout << "pow_backword()" << std::endl; break;  \
  }                                                                       \
}
  if (curr_tensor.tensor_data->visited || curr_tensor.tensor_data->parents.size() == 0) 
    return;
  curr_tensor.tensor_data->visited = true;
  GRAD_OpType();
//#define DEBUGING_MODE
#ifdef DEBUGING_MODE
  GRAD_FN();
#endif
  for (tensor& parent: curr_tensor.tensor_data->parents)
    backword(parent, parent.tensor_data->grad);
#undef GRAD_OpType
#undef GRAD_FN
}

void tensor::backword() {
  tensor_data->grad = af::constant(1, data().dims());
  backword(*this, tensor_data->grad);
}

} // namespace falcon