# Falcon : A Deeplearning Framework.

Falcon has a PyTorch-like API that supports C++. Contribution to Python support is welcome.

## Example  
PyTorch example.
```python
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
```

Falcon example. 
```c++
#include <arrayfire.h>
#include "falcon/falcon.h"

using namespace Falcon;

Tensor a = Tensor({2, 3}, true);
Tensor b = Tensor({6, 4}, true);
Tensor c = (a^3) * 3 - b^2;

c.backward();
```

## For Contributors 
```shell
git clone https://github.com/lilanka/Falcon.git
cd Falcon
mkdir build && cd build
cmake ..
make 
./falcon
```
