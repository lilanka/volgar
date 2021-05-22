#include "falcon/autograd/Functional.h"

Tensor F::add(const Tensor& a, const Tensor& b) {
  return Tensor(a.array() + b.array(), {a, b}, a.isGradOn(&b), 0);
}

Tensor F::div(const Tensor& a, const float b) {
  return Tensor(a.array() / b, {a}, a.isGradOn(nullptr), b, 2);
}

Tensor F::matmul(const Tensor& a, const Tensor& b) {
  return Tensor(af::matmul(a.array(), b.array()), {a, b}, a.isGradOn(&b), 4);
}

Tensor F::mul0(const Tensor& a, const float b) {
  return Tensor(a.array() * b, {a}, a.isGradOn(nullptr), b, 3);
}

Tensor F::mul1(const Tensor& a, const Tensor& b) {
  return Tensor(a.array() * b.array(), {a, b}, a.isGradOn(&b), 6);
}

Tensor F::sub(const Tensor& a, const Tensor& b) {
  return Tensor(a.array() - b.array(), {a, b}, a.isGradOn(&b), 1);
}

Tensor F::relu(const Tensor& a) {
  return Tensor(af::max(a.array(), af::constant(0, a.array().dims())), {a}, a.isGradOn(nullptr), 7);
}

Tensor F::sigmoid(const Tensor& a) {
  af::array ones_ = af::constant(1, a.array().dims());
  af::array sig_ =  ones_ / (ones_ + af::exp((-1) * a.array()));
  return Tensor(sig_, {a}, a.isGradOn(nullptr), 8);
}
