#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace Falcon;

int main() {

  af::array a = af::constant(1, 5, 5);
  af::array b = af::constant(2, 5, 5);
  af::array c = af::constant(3, 5, 10);
  af::array d = af::constant(4, 6, 5);
  
  Tensor x1 = Tensor(a, true);
  Tensor x2 = Tensor(b, true);
  Tensor x3 = Tensor(c, false);
  Tensor x4 = Tensor(d, true);

  Tensor y1 = x1 + x2 + (x2 - x1)* 4.5;
  Tensor y2 = x1 - x2;
  Tensor y3 = x1 * 2.5;
  Tensor y4 = x1 / 2.5;
  Tensor y5 = x1 * x2;
  Tensor add = y1 * 4.5 + y2;
  Tensor _dot = x4.matmul(x3);  // dot product

  Tensor add_dot = x1 + x2 - (x1.matmul(x2));

  F m;
  Tensor p = m.add(x1, x2);
  
  af_print(p.array());
  af_print(y1.array());
  af_print(y2.array());
  af_print(y3.array());
  af_print(y4.array());
  af_print(y5.array());

  af_print(add.array());
  af_print(_dot.array());
  af_print(add_dot.array());

  _dot.backward(af::constant(1, 6, 10));  
  _dot.gradFn();

  _dot.backward();  
  _dot.gradFn();

  add_dot.backward();  
  add_dot.gradFn();

  y5.backward();  
  y5.gradFn();

  af::array f = {0, 1, 2, 3, 4}; 
  Tensor x = Tensor(f, true);
  Tensor y = x * x;
  y.backward();
  y.grad();
}
