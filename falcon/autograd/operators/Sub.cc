#include "Sub.h"

Tensor Sub::forward(const Tensor& a, const Tensor& b) { 
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a, b});
  return Tensor(a.array() - b.array(), parents, a.isGradOn(&b));
}
