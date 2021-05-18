#include "Mul.h"

Tensor Mul::forward(const Tensor& a, const float b) { 
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a});
  return Tensor(a.array() * b, parents, a.isGradOn(nullptr));
}
