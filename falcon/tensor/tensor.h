#pragma once

#include <memory>
#include <string>
#include <vector>
#include <initializer_list>

#include <arrayfire.h>

#include "falcon/common.h"

namespace falcon {

// These are the ops that must be implemented
enum class OpType {
  // Binary Ops
  ADD, SUB, DIV, MUL, POW,
  // Unary Ops
  NEG, RELU, EXP, LOG,
  // Reduce Ops
  SUM, MAX
};

// Tensor constructor 
// The default data type is float (f32)
class tensor {
public:
  tensor() = default;
  tensor(af::array&& data, const bool requires_grad);
  tensor(af::array&& data, const bool requires_grad, std::vector<tensor> parents, OpType op);
  tensor(af::array&& data, const bool requires_grad, std::vector<tensor> parents, float mul, OpType op);

  // Get corresponding arrayfire tensor
  af::array& data() const;

  // Get the gradient
  af::array& grad() const;

  // Tensor linear operations
  tensor operator*(const float number);
  tensor operator*(const tensor& other);
  tensor operator/(const float number);
  tensor operator^(const float number);
  tensor operator+(const tensor& other);
  tensor operator-(const tensor& other);

  // Is gradient activated
  bool is_grad_on() const;

  /*
  friend std::ostream& operator<<(std::ostream& o, const tensor& x) {
      return o << "Tensor: \n" << af_print(x.data());
  }
  */

  // Gradients of operations 
  void add_backword(const af::array& output_grad) const;
  void sub_backword(const af::array& output_grad) const;
  void mul_backword(const af::array& output_grad) const;
  void div_backword(const af::array& output_grad) const;
  void pow_backword(const af::array& output_grad) const;

  // DFS algorithm
  // Calculate the gradients
  void backword(const tensor& curr_tensor, const af::array& output_grad);

  // Backward differentiation
  // TODO: Add higher order gradients
  // retain_graph for second order gradients
  void backword(bool retain_graph = false);
  
private:
  struct tensorData {
    af::array data;
    bool requires_grad{false};
    bool visited{false};
    af::array grad;               // Gradient of the output w.r.t grad
    float mul;                    // Used when operations uses scalars
    OpType op;                        // Opration type
    std::vector<tensor> parents;
  };
  
  // Tensor data
  std::shared_ptr<tensorData> tensor_data{std::make_shared<tensorData>()};
};

} // namespace falcon