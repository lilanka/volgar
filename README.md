# Falcon : A Deeplearning Framework.

Similar to the PyTorch. Currently only supports for C++. 

## Modules  
```shell
falcon/autograd = Computes gradients of the functions

**TODO:**
falcon/optim 
falcon/nn
falcon/Conv
```

## Example  
```c++
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace Falcon;

af::array a = af::constant(1, 5, 5);
af::array b = af::constant(1, 5, 5);
af::array c = af::constant(1, 5, 10);

Tensor x1 = Tensor(a, true);
Tensor x2 = Tensor(b, true);
Tensor x3 = Tensor(c, false);

Tensor y1 = x1 + x2 + (x2 - x1) * 4.5;

af_print(y1.array());

y1.backward(); // Backward propagation
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
