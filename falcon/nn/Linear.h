#pragma once

#include <arrayfire.h> 
#include <memory>
#include <vector>

#include "falcon/tensor/Tensor.h"

namespace Falcon {

class Linear {
public:
  /*
  *   set up the linear layer
  */
  Linear(int in_, int out_, bool bias_);  

  /*
  * oprations inside the node
  */
  Tensor operator()(const Tensor& inputs); 

  /*
  *  Outputs the weights and bias of the layer  
  */
  Tensor weights();
  Tensor bias();

private:
  struct layerData {
    std::unique_ptr<Tensor> bias{nullptr};              // bias 
    std::unique_ptr<Tensor> params{nullptr};            // weights initialized 
  };
  std::shared_ptr<layerData> layerData_{std::make_shared<layerData>()};
};
} // namespace falcon
