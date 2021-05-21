#include "falcon/autograd/Functional.h"

Tensor F::add(const Tensor& a, const Tensor& b) {
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a, b});
  return Tensor(a.array() + b.array(), parents, a.isGradOn(&b), 0);
}

Tensor F::div(const Tensor& a, const float b) {
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a});
  return Tensor(a.array() / b, parents, a.isGradOn(nullptr), b, 2);
}

Tensor F::matmul(const Tensor& a, const Tensor& b) {
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a, b});
  return Tensor(af::matmul(a.array(), b.array()), parents, a.isGradOn(&b), 4);
}

Tensor F::mul0(const Tensor& a, const float b) {
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a});
  return Tensor(a.array() * b, parents, a.isGradOn(nullptr), b, 3);
}

Tensor F::mul1(const Tensor& a, const Tensor& b) {
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a, b});
  return Tensor(a.array() * b.array(), parents, a.isGradOn(&b), 6);
}

Tensor F::sub(const Tensor& a, const Tensor& b) {
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a, b});
  return Tensor(a.array() - b.array(), parents, a.isGradOn(&b), 1);
}
