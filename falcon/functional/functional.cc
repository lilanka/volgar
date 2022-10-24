#include <arrayfire.h>

#include "functional.h"

namespace falcon {

// TODO: Check dimensions before calculations

tensor F::add(const tensor& x, const tensor& y) {
  return tensor(x.data() + y.data(), x.is_grad_on() || y.is_grad_on(), {x, y}, OpType::ADD);
}

tensor F::sub(const tensor& x, const tensor& y) {
  return tensor(x.data() - y.data(), x.is_grad_on() || y.is_grad_on(), {x, y}, OpType::SUB);
}

tensor F::div(const tensor& x, const float number) {
  return tensor(x.data() / number, x.is_grad_on(), {x}, number, OpType::DIV);
}

tensor F::mul(const tensor& x, const float number) {
  return tensor(x.data() * number, x.is_grad_on(), {x}, number, OpType::MUL);
}

tensor F::mul(const tensor& x, const tensor& y) {
  return tensor(x.data() * y.data(), x.is_grad_on(), {x, y}, OpType::MUL);
}

tensor F::pow(const tensor& x, const float number) {
  return tensor(af::pow(x.data(), number), x.is_grad_on(), {x}, number, OpType::POW);
}

} // namespace falcon