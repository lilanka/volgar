from volgar.tensor import Tensor
import volgar.nn.functional as F

x = Tensor([1, 2], requires_grad=True)
y = Tensor([1, 2], requires_grad=True)

z = F.add(x, y)
z()