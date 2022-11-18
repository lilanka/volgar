from volgar.tensor import Tensor
import volgar.nn as vnn

import numpy as np
import torch
import torch.nn as nn

def test_relu():
  a = Tensor([2, 3], requires_grad=True)

  # volgar test
  relu = vnn.ReLU()
  y = relu(a)

  # torch test
  torch_a = torch.tensor(a.data, dtype=torch.float32)

  relu = nn.ReLU()
  torch_y = relu(torch_a)

  # testing
  np.testing.assert_allclose(y.data, torch_y.detach().numpy(), atol=5e-4, rtol=1e-5)

def test_tanh():
  a = Tensor([2, 3], requires_grad=True)

  # volgar test
  tanh = vnn.Tanh()
  y = tanh(a)

  # torch test
  torch_a = torch.tensor(a.data, dtype=torch.float32)

  tanh = nn.Tanh()
  torch_y = tanh(torch_a)

  # testing
  np.testing.assert_allclose(y.data, torch_y.detach().numpy(), atol=5e-4, rtol=1e-5)

def test_sigmoid():
  a = Tensor([2, 3], requires_grad=True)

  # volgar test
  sigmoid = vnn.Sigmoid()
  y = sigmoid(a)

  # torch test
  torch_a = torch.tensor(a.data, dtype=torch.float32)

  sigmoid = nn.Sigmoid()
  torch_y = sigmoid(torch_a)

  # testing
  print(y, torch_y)
  np.testing.assert_allclose(y.data, torch_y.detach().numpy(), atol=5e-4, rtol=1e-5)

#test_sigmoid()
#test_tanh()
test_relu()