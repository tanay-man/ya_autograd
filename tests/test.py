from tensor import *

a = np.array([3])
b = np.array([2])

a = Tensor(a, requires_grad=True)
b = Tensor(b)

c = a * a + b
c.backward()
print(c)
print(a)
print(b)