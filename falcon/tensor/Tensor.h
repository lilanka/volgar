/*
/ todo:
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
  Tensor(af::array data, std::vector<Tensor> parents, bool requires_data, int _op);

  /*
  * constructor with op: grad function , mul: use when operations with scalar values
  */
  Tensor(af::array data, std::vector<Tensor> parents, bool requires_data, float _mul, int _op);

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
  * element wise multiplication 
  */
  Tensor operator*(const float num);

  /*
  * element wise multiplication of two tensors 
  */
  Tensor operator*(const Tensor& tensor);
  /*
  * divide input tensor and a number
  */
  Tensor operator/(const float num); 

  /*
  * Taking power of a tensor
  */
  Tensor operator^(const float num);

  /*
  * dot product of two tensors
  */
  Tensor matmul(const Tensor& tensor);


  
  // check whether the output tensor should requires_grad on or not
  bool isGradOn(const Tensor* other) const;
 
  /*
  * DFS
  */
  void backward(const Tensor& tensor, const af::array& output_grad);  

  /*
  * kik off the backward pass 
  */
  void backward(af::array initial_grad);

  void backward();
  // show the grad 
  af::array grad() const; 

  /*
  * gradient calculation functions in each operation
  */
  void addBackward(const af::array& output_grad) const;
  void subBackward(const af::array& output_grad) const;
  void divBackward(const af::array& output_grad) const;
  void mulBackward0(const af::array& output_grad) const;
  void matmulBackward(const af::array& output_grad) const; 
  void mulBackward1(const af::array& output_grad) const;
  void powBackward(const af::array& output_grad) const;
  
  // for activation functions
  void reluBackward(const af::array& output_grad) const;
  void sigmoidBackward(const af::array& output_grad) const;
  void tanhBackward(const af::array& output_grad) const;

  // shows the nodes meaning (correspodning operation in the node)
  void gradFn(); 

  /*
  * used for backward operation. call the backward functions of each operation
  */
  void gradOp(const Tensor& tensor, const af::array& output_grad);  
private:   
  /*
  * tensor data stored in here
  */ 
  struct tensorData {
    af::array data;                       // data of the variable
    bool requires_grad{false};            // does this variable calculate the grads
    std::vector<Tensor> parents;          // parents of this variable
    std::unique_ptr<af::array> grad{nullptr}; // gradient of output w.r.t this tensor 
    bool visited{false};                  // marked as visited in DFS
    std::unique_ptr<float> _mul{nullptr}; // used when multiplication or pow with scalar
    std::unique_ptr<int> _op{nullptr};    // operation (gradFn) 
  }; 

  std::shared_ptr<tensorData> tensorData_{std::make_shared<tensorData>()};
};
}
