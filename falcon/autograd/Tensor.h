/*
  todo:
    hash table is more computationaly efficient 
    current use linked lists. use adjescent list with hash tables  
*/

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
 

  void backward(const Tensor& tensor, const af::array& output_grad);  
  /*
  * kik off the backward pass 
  */
  void backward();
  
  void grad() const;

  void addBackward(const af::array& output_grad);
private:   
  /*
  * tensor data stored in here
  */ 
  struct tensorData {
    af::array data;  // data of the variable
    bool requires_grad{false}; // does this variable calculate the grads
    std::vector<Tensor> parents; // parents of this variable
    std::vector<af::array> grad; // gradient of the variable
    bool visited{false};
  }; 

  std::shared_ptr<tensorData> tensorData_{std::make_shared<tensorData>()};
};
}
