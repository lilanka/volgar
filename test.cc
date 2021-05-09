#include <iostream>
#include <arrayfire.h>

#include "falcon.h"

using namespace Falcon;

int main() {
  af::array a = af::constant(1, 5, 5);
  af::array b = af::constant(5, 5, 5);
  Tensor x1 = Tensor(a, true);
  Tensor x2 = Tensor(b, true);

  Tensor x_out = x1 + x2;
}
