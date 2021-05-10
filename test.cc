#include <iostream>
#include <arrayfire.h>

#include "falcon.h"

using namespace Falcon;

int main() {
  af::array a = af::constant(1, 5, 5);
  af::array b = af::constant(5, 5, 5);

  Tensor x1 = Tensor(a, true);
  Tensor x2 = Tensor(b, true);

  Tensor y1 = x1 + x2;
  Tensor y2 = x1 - x2;
  Tensor y3 = x1 * 2.5;
  Tensor y4 = x1 / 2.5;
  Tensor _add = y1 + y2;

  af_print(y1.array());
  af_print(y2.array());
  af_print(y3.array());
  af_print(y4.array());

  af_print(_add.array());
}
