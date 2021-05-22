// This is a test file of development. A serious, series of mess :)

#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace Falcon;

int main() {
  Tensor aa = Tensor({2, 3}, true);
  Tensor bb = Tensor({6, 4}, true);

  Tensor Q = aa*aa*aa*3 - bb*bb;
  Tensor check_Q = aa*aa*9;

  Q.backward({1, 1});
  af_print(aa.grad());
  af_print(check_Q.array());
}
