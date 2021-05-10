#include <arrayfire.h>

class Mul {
public:
  Mul() = default;
  af::array forward(const af::array& a, const float b);
};
