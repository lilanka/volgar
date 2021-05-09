#include <arrayfire.h>

namespace Falcon {

class Tensor {
public:
  Tensor() = default;

  Tensor(af::array& data, bool requires_grad): 
    data{data}, requires_grad{requires_grad} {}

  Tensor operator+(const Tensor& tensor);
  Tensor operator-(const Tensor& tensor);

  // check whether the output tensor should requires_grad on or not
  bool isGradOn(bool val);

private:   
  af::array& data;
  bool requires_grad; 
};

}
