// This is a test file of development. A serious, series of mess :)

#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace Falcon;

int main() {
  Tensor aa = Tensor({2, 3}, true);
  Tensor bb = Tensor(af::constant(2, 2, 10), true);
  Tensor cc = Tensor(af::constant(1, 5, 2), false);

  //Tensor Q = (aa^3)*3 - (bb^2);
  Tensor Q = cc.matmul(bb); 
  Q.backward();

  af_print(Q.array());
  af_print(cc.grad());
}
