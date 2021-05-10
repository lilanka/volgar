#include <arrayfire.h>

class Div {
public:
  Div() = default;
  af::array forward(const af::array& a, const float b);
};
