# Falcon : A Deeplearning Framework.

Similar to the PyTorch. Currently only supports for C++. 

## Modules  
```
falcon/autograd = Computes gradients of the functions

**TODO:**
falcon/optim 
falcon/nn
falcon/Conv
```

## Example  
```
#include <arrayfire.h>

#include "falcon/falcon.h"

using namespace Falcon;

af::array a = af::constant(1, 5, 5);
af::array b = af::constant(1, 5, 5);
af::array c = af::constant(1, 5, 10);

Tensor x1 = Tensor(a, true);
Tensor x2 = Tensor(b, true);
Tensor x3 = Tensor(c, false);

Tensor y1 = x1 + x2;
Tensor y2 = x1 - x2;
Tensor add = y1 + y2;

af_print(y1.array());
af_print(y2.array());

af_print(add.array());

add.backward(); // Backward propagation
```
