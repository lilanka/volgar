#pragma once 

#include <arrayfire.h>
#include <utility>
#include <memory>
#include <string>
#include <vector>

namespace Falcon {

class Tensor {
public:
  /*
  * default constructor
  */
  Tensor() = default;

  /*
  * construct a tensor using arrayfire 
  */
  Tensor(af::array data, bool requires_grad);  

  /*
  * for inside tensor genration
  */
  Tensor(af::array data, std::vector<Tensor> parents, bool requires_data);

  /*
  * output arrayfire array
  */
  af::array& array() const;
 
  /*
  * set up parents
  */
  std::vector<Tensor> parentSetUp(const Tensor* other);  

  /*
  * adds two input tensors
  */
  Tensor operator+(const Tensor& tensor);

  /*
  * substitute two input tensors
  */
  Tensor operator-(const Tensor& tensor);

  /*
  * multiply input tensor and a number
  */
  Tensor operator*(const float num);

  /*
  * divide input tensor and a number
  */
  Tensor operator/(const float num); 

  /*
  * dot product of two tensors
  */
  Tensor matmul(const Tensor& tensor);
  
  // check whether the output tensor should requires_grad on or not
  bool isGradOn(const Tensor* other) const;
  
  /*
  * backward pass
  */
  void backward();

private:   
  /*
  * tensor data stored in here
  */ 
  struct tensorData {
    af::array data;  // data of the variable
    bool requires_grad{false}; // does this variable calculate the grads
    std::vector<Tensor> parents; // parents of this variable
    std::vector<af::array> grad; // gradient of the variable
  }; 

  std::shared_ptr<tensorData> tensorData_{std::make_shared<tensorData>()};
};
}
