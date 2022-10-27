#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace falcon;

static tensor rosenbrock_banana(tensor& x1, tensor& x2) {
  tensor ones = tensor(af::constant(1, x1.data().dims()), false);
  tensor fives = tensor(af::constant(5, x1.data().dims()), false);

  return ((ones - x1)^2) + fives * ((x2 - x1^2)^2);
}

int main() {
  tensor x1 = tensor({1}, true);
  tensor x2 = tensor({1}, true);

  tensor rb = rosenbrock_banana(x1, x2);

  // calculate gradients
  rb.backward();

  af_print(x1.grad());
  af_print(x2.grad());
}