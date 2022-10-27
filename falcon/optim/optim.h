#pragma once

#include <vector>

#include <arrayfire.h>
#include "falcon/tensor/tensor.h"

namespace falcon {

// Falcon optimization library
class Optim {
public:
  optim() = default;  

  // Unvaraite function for testing
  tensor test_fx(const tensor& x) {
    tensor fx = x**2 + x * 2 + tensor(af::constant(1, x.data().dims()), false);
    return fx;
  }

protected:
  // Find an initial bracket
  std::vector<float> init_brackets(const float dis, const float ss) {
    tensor a = tensor({0}, false);
    tensor ya = test_fx(a);
    tensor s = tensor({dis}, false);
    tensor b = a + s;
    tensor yb = test_fx(b);

    tensor step_size = tensor({step_size}, false);

    if (yb > ya) {
      tensor temp_a = a;
      tensor temp_ya = ya;
      a = b;
      b = a;
      ya = yb;
      yb = ya;
      s = -s;
    }

    while (true) {
      tensor c = b + s;
      tensor yc = test_fx(c);

      if (yc > yb) 
        return {a, c} if (a < c) else {c, a};
      a = b;
      ya = yb;
      b = c;
      yb = ys
      s *= step_size;
    }
  }
};

} // namespace falcon