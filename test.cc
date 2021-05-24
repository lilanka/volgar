// This is a test file of development. A serious, series of mess :)

#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace Falcon;

int main() {
  Tensor aa = Tensor({2, 3}, true);
  Tensor bb = Tensor({6, 4}, true);

  Tensor Q = (aa^3)*3 - (bb^2);
  Q.backward({1, 1});

  af_print(aa.grad());
  af_print(bb.grad());
}
