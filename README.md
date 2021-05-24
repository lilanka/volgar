# Falcon : A Deeplearning Framework.

Similar to the PyTorch. Currently only supports for C++. 

## Modules  
```shell
falcon/autograd = Computes gradients of the functions

**TODO**
falcon/optim 
falcon/nn
falcon/Conv
```

## Example  
This is how PyTorch does it
```python
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
```

This is how Falcon does it
```c++
#include <arrayfire.h>
#include "falcon/falcon.h"

using namespace Falcon;

Tensor aa = Tensor({2, 3}, true);
Tensor bb = Tensor({6, 4}, true);

Tensor Q = (aa^3)*3 - (bb^2);

Q.backward({1, 1});
```

## Test 
```shell
git clone https://github.com/lilanka/Falcon.git
cd Falcon
mkdir build && cd build
cmake ..
make 
./falcon
```
