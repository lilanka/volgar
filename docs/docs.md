# Falcon Documentation

## Tensors
**falcon.tensor(data, requires_grad=false)**  
Constructing tensors with no autograd ho
Tensors can be expressed as 
```c++
tensor a = tensor()
tensor b = tensor({1, 3}, true)
```

* https://pytorch.org/docs/stable/notes/autograd.html
* https://arxiv.org/pdf/1502.05767.pdf
* https://pytorch.org/docs/stable/generated/torch.tensor.html

## Optim
* Bracketing: Find an interval which derivative is positive and negetive from the opposit
direction from a point.
* Reduce the bracket size.
  - Fibonacci search
  - Golden section search: Uses the golden ratio to approximate Fibonacci search
  - Guadratic fit search
    - Faster than Golden section search
  - Shubert-Piyovskii method: Global optimization method over a domain [a, b].
    - Guaranteed to converge on the global minimum of a function irrespective of any local minima
    or whether the function is unimodel
* To ensure resulting method finds a local minima use Bisection Method. It find roots of a 
function.
* Large step factor (lr) will result in faster convergence but risk of overshooting the 
minimum. Smaller lr can be more stable but slover convergence.   
* Decaying step factor is usefull for minimizing noisy objective functions.