#include "Mul.h"

Tensor Mul::forward(const Tensor& a, const float b) { 
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a});
  return Tensor(a.array() * b, parents, a.isGradOn(nullptr), b, 3);
}

Tensor Mul::forward(const Tensor& a, const Tensor& b) { 
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a, b});
  return Tensor(a.array() * b.array(), parents, a.isGradOn(&b), 6);
}
