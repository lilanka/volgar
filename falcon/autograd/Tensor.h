#pragma once

#include <arrayfire.h>
#include <utility>
#include <memory>

#include "Add.h"
#include "Sub.h"
#include "Mul.h"
#include "Div.h"

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
  * output arrayfire array
  */
  af::array& array() const;
  
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
  
  // check whether the output tensor should requires_grad on or not
  bool isGradOn(bool val);

private:   
  /*
  * tensor data stored in here
  */ 
  struct tensorData {
    af::array data; 
    bool requires_grad{false};
  }; 

  std::shared_ptr<tensorData> tensorData_{std::make_shared<tensorData>()};

  Add add;
  Sub sub;
  Mul mul;
  Div div;
};
}
