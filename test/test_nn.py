import torch
import numpy as np

from volgar.tensor import Tensor
import volgar.nn as vnn 

def test_conv2d():
  model = vnn.Conv2d(6, 10, 3) 

  x = np.random.randn(5, 6, 32, 16)
  x = Tensor(x)
  print(x.size())

def test_linear():
  x = Tensor([[1, 2, 3, 4, 5]])

  # volgar model
  linear = vnn.Linear(5, 6)
  vrelu = vnn.ReLU() 
  y = vrelu(linear(x))

  # same thing in torch
  with torch.no_grad():
    torch_layer = torch.nn.Linear(5, 6).eval()
    torch_layer.weight[:] = torch.tensor(linear.weight.data, dtype=torch.float32)
    torch_layer.bias[:] = torch.tensor(linear.bias.data, dtype=torch.float32)
    torch_x = torch.tensor(x.data, dtype=torch.float32)
    torch_relu = torch.nn.ReLU()
    torch_y = torch_relu(torch_layer(torch_x))

  # testing
  np.testing.assert_allclose(y.data, torch_y.detach().numpy(), atol=5e-4, rtol=1e-5)

  """
  k = vrelu(x)
  k.backward()
  print(x.grad)
  """

test_linear()
#test_conv2d()