#include "falcon/autograd/operators/Div.h"

Tensor Div::forward(const Tensor& a, const float b) { 
  return Tensor(a.array() / b, a.isGradOn(nullptr)); 
}
