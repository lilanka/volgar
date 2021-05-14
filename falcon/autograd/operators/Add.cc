#include "falcon/autograd/operators/Add.h"

Tensor Add::forward(const Tensor& a, const Tensor& b) { 
  return Tensor(a.array() + b.array(), a.isGradOn(&b)); 
}
