#include "Matmul.h"

Tensor Matmul::forward(const Tensor& a, const Tensor& b) {
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a, b});
  return Tensor(af::matmulTN(a.array(), b.array()), parents, a.isGradOn(&b), 4);
}
