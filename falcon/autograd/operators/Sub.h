#include <arrayfire.h>

class Sub {
public:
  Sub() = default;
  af::array forward(const af::array& a, const af::array& b);
};
