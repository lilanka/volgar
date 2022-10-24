#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace falcon;

int main() {
  tensor A = tensor();

  tensor B = tensor({2, 3}, true);
  tensor C = tensor({4, 5}, true);
  tensor D = tensor({{1, 3}, {5, 6}}, true);

  tensor added = B + C;
  tensor divided = B / 2;
  tensor multiplied = B * 2;
  tensor power = B^2;
  tensor ten_added = D + D;

  af_print(B.data());
  af_print(C.data());

  return 0;
}