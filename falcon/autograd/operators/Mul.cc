#include "Mul.h"

Tensor Mul::forward(const Tensor& a, const float b) { 
  return Tensor(a.array() * b, a.isGradOn(nullptr));
}
