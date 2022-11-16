# Volgar - A framework for scientific computing.
The framework is for serious computing purposes (i.e university assignments). 
If it doesn't support your needs, add it and use it.
<br><br>
This is the kind of first principle method of learning AI.

## Autodiff
PyTorch like automatic differentiation library.

```python
from volgar.tensor import Tensor 

a = Tensor([2, 3], requires_grad=True)
b = Tensor([6, 4], requires_grad=True)

c = (a ** 3) * 3 - b ** 2
c.backward()
```

## Neural Networks
Yeah, similar to PyTorch
```python
x = Tensor([[1, 2, 3, 4, 5]])

# Linear layers
model = Linear(5, 6)
y = model(x)
```