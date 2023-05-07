# Volgar - Deep Learning Framework
This is the kind of first principle method of learning AI. Implemented reverse mode autodiff and neural networks.

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
