#include <arrayfire.h>

class Matmul {
public:
  Matmul() = default;
  af::array forward(const af::array& a, const af::array& b);
};
