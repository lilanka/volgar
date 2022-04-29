## Falcon Documentation

### Example
```c++
#include "falcon/falcon.h"

using namespace falcon;

// Tensor initialization
Tensor a = Tensor({2, 3}, true);
Tensor b = Tensor({4, 5}, true);

Tensor addition = a + b;
```

### Print tensor
```c++
#include <arrayfire.h>

Tensor a = Tensor({2, 3}, true);
af_print(a.array());
```
