#include "falcon/tensor/Tensor.h"

using namespace Tensor;
namespace Falcon {

class Linear {
public:
  /*
  *   set up the linear layer
  */
  Linear(int in_nodes, int out_nodes, bool bias);  

  /*
  *  Outputs the parameters of the layer  
  */
  Tensor parameters();
};
}
