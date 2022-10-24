#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace falcon;

int main() {
  /*
  tensor a = tensor({2, 3}, true);
  tensor b = tensor({6, 4}, true);

  tensor c = ((a^3)*3) - (b^2);
  c.backword();

  af_print(a.grad());
  af_print(((a^2)*9).data());
  af_print(b.grad());
  af_print((b*(-2)).data());
  */

  tensor x = tensor({2, 3}, true);
  tensor y = tensor({5, 6}, true);
  tensor z = tensor({7, 8}, true);

  tensor q = ((x^9)*5) - y - (z^5);
  q.backword();

  af_print(x.grad());
  af_print(((x^8)*5*9).data());
  af_print(y.grad());
  af_print(z.grad());
  af_print(((z^4)*(-5)).data());
}