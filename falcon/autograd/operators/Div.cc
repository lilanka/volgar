#include "falcon/autograd/operators/Div.h"

Tensor Div::forward(const Tensor& a, const float b) { 
  std::vector<Tensor> parents;
  parents.insert(parents.end(), {a});
  return Tensor(a.array() / b, parents, a.isGradOn(nullptr)); 
}
