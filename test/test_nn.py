import torch
import numpy as np

from volgar.tensor import Tensor
from volgar.nn import Linear, Conv2d

def test_conv2d():
  model = Conv2d(6, 10, 3) 

def test_linear():
  x = Tensor([[1, 2, 3, 4, 5]])

  # volgar model
  model = Linear(5, 6)
  y = model(x)

  print(y)

  # same thing in torch
  with torch.no_grad():
    torch_layer = torch.nn.Linear(5, 6).eval()
    torch_layer.weight[:] = torch.tensor(model.weight.data, dtype=torch.float32)
    torch_layer.bias[:] = torch.tensor(model.bias.data, dtype=torch.float32)
    torch_x = torch.tensor(x.data, dtype=torch.float32)
    torch_y = torch_layer(torch_x)

  # testing
  np.testing.assert_allclose(y.data, torch_y.detach().numpy(), atol=5e-4, rtol=1e-5)

#test_linear()
test_linear()