#include "Sub.h"

Tensor Sub::forward(const Tensor& a, const Tensor& b) { 
  return Tensor(a.array() - b.array(), a.isGradOn(&b));
}
