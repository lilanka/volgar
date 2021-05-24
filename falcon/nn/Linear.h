#include <arrayfire.h> 

#include "falcon/tensor/Tensor.h"

using namespace Tensor;
namespace Falcon {

class Linear {
public:
  /*
  * setup layer bias = false
  */
  Linear(int in_, out_);

  /*
  *   set up the linear layer
  */
  Linear(int in_, int out_, bool bias);  

  /*
  *  Outputs the parameters of the layer  
  */
  Tensor parameters();

private:
  bool bias{false};
  std::vector<af::array<af::array>> Jocob;
};
}
