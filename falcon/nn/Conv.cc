#include "falcon/nn/Conv.h"
#include "falcon/utils.h"
#include "falcon/tensor/Distributions.h"

namespace Falcon {

ConvND::Conv2D(int in_channels, int out_channels, std::vector<int> kernel_size, std::vector<int> stride std::vector<int> padding, bool bias) {

  _C = kernel_size[0]; 
  _N = kernel_size[1]; 

  // define kernals
  for (int i=0; i < out_channels; i++) {
    std::unique_ptr<Tensor> _kernal{nullptr};
    while (true) {
      _c = randomInt(1, _C); 
      _n = randomInt(1, _N);
      if out_channels == _c*_n^2 {
        _kernal = std::make_unique<Tensor>(ones(_C-_c+1, _N-_n+1, _N-_n+1));
        kernels._cat(_kernal);  
      }
    }  
  }
}

Tensor Conv2D::operator()(const Tensor& input) {
   
}
} // namespace falcon
