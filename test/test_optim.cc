#include <iostream>
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace falcon;

int main() {
  Optim optim;

  tensor x = tensor({1}, true);
  tensor fx = optim.test_fx(x);

}