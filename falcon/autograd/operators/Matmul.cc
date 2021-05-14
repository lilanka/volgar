#include "Matmul.h"

Tensor Matmul::forward(const Tensor& a, const Tensor& b) {
  return Tensor(af::matmulTN(a.array(), b.array()), a.isGradOn(&b));
}
