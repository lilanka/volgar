#include <arrayfire.h>

class Add {
public:
  Add() = default;
  af::array forward(const af::array& a, const af::array& b);    
};
