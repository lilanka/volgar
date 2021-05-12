#include "Matmul.h"

af::array Matmul::forward(const af::array& a, const af::array& b) {
  return af::matmulTN(a, b);
}
