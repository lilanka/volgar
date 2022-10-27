#include <iostream>

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

tensor tensor::operator*(const tensor& other) {
  F f;
  return f.mul(*this, other);
}

tensor tensor::operator^(const float number) {
  F f;
  return f.pow(*this, number);
}

void tensor::add_backward(const af::array& output_grad) const {
  for (tensor& parent: tensor_data->parents) {
    if (parent.tensor_data->requires_grad)
      parent.tensor_data->grad += output_grad;
  }
}

void tensor::sub_backward(const af::array& output_grad) const {
  if (tensor_data->parents[0].tensor_data->requires_grad)
    tensor_data->parents[0].tensor_data->grad += output_grad;
  if (tensor_data->parents[1].tensor_data->requires_grad)
    tensor_data->parents[1].tensor_data->grad -= output_grad;
}

void tensor::mul_backward(const af::array& output_grad) const {
  if (tensor_data->parents.size() == 1)
    tensor_data->parents[0].tensor_data->grad += output_grad * tensor_data->mul;  
  else {
    std::vector<tensor> parents = tensor_data->parents;
    if (tensor_data->parents[0].tensor_data->requires_grad)
      tensor_data->parents[0].tensor_data->grad += parents[1].data() * output_grad;
    if (tensor_data->parents[1].tensor_data->requires_grad)
      tensor_data->parents[1].tensor_data->grad += parents[0].data() * output_grad;
  }
}

void tensor::div_backward(const af::array& output_grad) const {
  if (tensor_data->parents[0].tensor_data->requires_grad)
    tensor_data->parents[0].tensor_data->grad += output_grad / tensor_data->mul;
}

void tensor::pow_backward(const af::array& output_grad) const {
  if (tensor_data->parents[0].tensor_data->requires_grad) {
    tensor_data->parents[0].tensor_data->grad += output_grad * tensor_data->mul * \
      af::pow(tensor_data->parents[0].data(), tensor_data->mul - 1);
  }
}

void tensor::sin_backward(const af::array& output_grad) const {
  if (tensor_data->parents[0].tensor_data->requires_grad)
    tensor_data->parents[0].tensor_data->grad += \
      af::cos(tensor_data->parents[0].tensor_data->data) * output_grad; 
}

void tensor::cos_backward(const af::array& output_grad) const {
  if (tensor_data->parents[0].tensor_data->requires_grad)
    tensor_data->parents[0].tensor_data->grad -= \
      af::sin(tensor_data->parents[0].tensor_data->data) * output_grad; 
}

void tensor::backward(const tensor& curr_tensor, const af::array& output_grad) {
#define GRAD_OPTYPE() {                                               \
  switch (curr_tensor.tensor_data->op) {                              \
    case OpType::ADD: curr_tensor.add_backward(output_grad); break;   \
    case OpType::SUB: curr_tensor.sub_backward(output_grad); break;   \
    case OpType::MUL: curr_tensor.mul_backward(output_grad); break;   \
    case OpType::DIV: curr_tensor.div_backward(output_grad); break;   \
    case OpType::POW: curr_tensor.pow_backward(output_grad); break;   \
    case OpType::SIN: curr_tensor.sin_backward(output_grad); break;   \
    case OpType::COS: curr_tensor.cos_backward(output_grad); break;   \
  }                                                                   \
}
#define GRAD_FN() {                                                       \
  switch (curr_tensor.tensor_data->op) {                                  \
    case OpType::ADD: std::cout << "add_backward()" << std::endl; break;  \
    case OpType::SUB: std::cout << "sub_backward()" << std::endl; break;  \
    case OpType::MUL: std::cout << "mul_backward()" << std::endl; break;  \
    case OpType::DIV: std::cout << "div_backward()" << std::endl; break;  \
    case OpType::POW: std::cout << "pow_backward()" << std::endl; break;  \
    case OpType::SIN: std::cout << "sin_backward()" << std::endl; break;  \
    case OpType::COS: std::cout << "cos_backward()" << std::endl; break;  \
  }                                                                       \
}
  if (curr_tensor.tensor_data->visited || curr_tensor.tensor_data->parents.size() == 0) 
    return;
  curr_tensor.tensor_data->visited = true;
  GRAD_OPTYPE();
//#define DEBUGING_MODE
#ifdef DEBUGING_MODE
  GRAD_FN();
#endif
  for (tensor& parent: curr_tensor.tensor_data->parents)
    if (parent.tensor_data->requires_grad)
      backward(parent, parent.tensor_data->grad);
#undef GRAD_OPTYPE
#undef GRAD_FN
}

void tensor::backward(bool retain_graph) {
  // Implicit gradient creation
  tensor_data->grad = af::constant(1, data().dims());
  backward(*this, tensor_data->grad);
}

} // namespace falcon