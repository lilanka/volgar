from volgar.tensor import Tensor 

a = Tensor([2, 3], requires_grad=True)
b = Tensor([6, 4], requires_grad=True)

c = (a ** 3) * 3 - b ** 2
c.backward()

print(a.grad)
print((a**2) * 9)
print(b.grad)
print(b * (-2))