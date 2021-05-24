// This is a test file of development. A serious, series of mess :)

#include <iostream>
#include <arrayfire.h>

#include "falcon/tensor/Tensor.h"
#include "falcon/tensor/Distributions.h"

using namespace Falcon;

int main() {
  Tensor aa = Tensor({2, 3}, true);
  Tensor bb = Tensor({4, 6}, true);
  
  Tensor Q = (aa^3)*3 - (bb^2);
  Q.backward();
  af_print(aa.grad());

  Tensor distro = normal(1, 3, true);
  af_print(distro.array());
}