#include <iostream>
#include <arrayfire.h>

#include "falcon/nn/Linear.h"

using namespace Falcon;

int main() {
  Linear l1 = Linear(4, 3, false);

  Tensor inputs = Tensor(af::randu(100, 4), false);
  Tensor l1_out = l1(inputs); 

  af_print(l1_out.array());

  Tensor param = l1.weights();
  Tensor bias = l1.bias();
  af_print(param.array());
  af_print(bias.array());
}
