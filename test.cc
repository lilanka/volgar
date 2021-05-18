#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace Falcon;

int main() {
  af::array a = af::constant(1, 5, 5);
  af::array b = af::constant(1, 5, 5);
  af::array c = af::constant(1, 5, 10);
  
  Tensor x1 = Tensor(a, true);
  Tensor x2 = Tensor(b, true);
  Tensor x3 = Tensor(c, false);
  
  Tensor y1 = x1 + x2;
  Tensor y2 = x1 - x2;
  Tensor y3 = x1 * 2.5;
  Tensor y4 = x1 / 2.5;
  Tensor add = y1 + y2;
  Tensor _dot_ = x1.matmul(x2);
  Tensor _dot = x2.matmul(x3);  // dot product

  Tensor add_dot = x1 + x2 - (x1.matmul(x2));

  af_print(y1.array());
  af_print(y2.array());
  af_print(y3.array());
  af_print(y4.array());

  af_print(add.array());
  af_print(_dot_.array());
  af_print(_dot.array());
  af_print(add_dot.array());

  add.backward();
}
