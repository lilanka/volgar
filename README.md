# Falcon : A Deeplearning Framework.

This project is to get deep understanding of field of AI. Currently only supports for C++. In development just out of curiosity. Use cases are similar to PyTorch DL framework. Some modules are in progress. 

This framework is depend on arrayfire tensor library. Hard to work with. Lot of dependancies. We shoould make our own. Look [nD](https://github.com/lilanka/nD)

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
or 
```
./run.sh
```
