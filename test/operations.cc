#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace Falcon;

int main() {
  Tensor a = Tensor({2, 3}, true);
  Tensor b = Tensor({6, 4}, true);
  
  Tensor Q = (aa^3)*3 - (bb^2);
  Q.backward();
  
  af_print(aa.grad());
}
