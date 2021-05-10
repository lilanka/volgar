#include <iostream>
#include <arrayfire.h>

#include "falcon.h"

using namespace Falcon;

int main() {
  af::array a = af::constant(1, 5, 5);
  af::array b = af::constant(5, 5, 5);

  Tensor x1 = Tensor(a, true);
  Tensor x2 = Tensor(b, true);

  Tensor x1_out = x1 + x2;
  Tensor x2_out = x1 - x2;
  Tensor x3_out = x1 * 2.5;
  Tensor x4_out = x1 / 2.5;

  af_print(x1_out.array());
  af_print(x2_out.array());
  af_print(x3_out.array());
  af_print(x4_out.array());

  Tensor _add = x1_out + x2_out;
  af_print(_add.array());
}
