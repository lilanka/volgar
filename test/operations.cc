// This is a test file of development. A serious, series of mess :)

#include <iostream>
#include <arrayfire.h>

#include "falcon/tensor/Tensor.h"
//#include "falcon/tensor/Distributions.h"
#include "falcon/autograd/Functional.h"

using namespace Falcon;

int main() {
  Tensor aa = Tensor({-10, -1, 0, 1, 2, 3}, true);
  Tensor bb = Tensor({4, 6}, true);
  
  //Tensor Q = (aa^3)*3 - (bb^2);
  //Q.backward();
  //af_print(aa.grad());
  F f; 
  Tensor x = f.tanh(aa);
  x.backward();
  af_print(aa.grad());

  //Tensor distro = normal(1, 3, true);
  //af_print(distro.array());
}
