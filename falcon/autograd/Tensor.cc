#include "Tensor.h"

namespace Falcon {

  bool Tensor::isGradOn(bool val) {
    if (requires_grad || val) 
     return true;
    return false; 
  }

  Tensor Tensor::operator+(const Tensor& tensor) {
    af::array d = data + tensor.data;
    Tensor x = Tensor(d, isGradOn(tensor.requires_grad));
    af_print(x.data);
    return tensor;
  }
}
