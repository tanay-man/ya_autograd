from src.tensor import *

a = np.array([3])
b = np.array([2])

a = Tensor(a, requires_grad=True)
b = Tensor(b)

# c = a * b
# print(c)

c = a * a + b

print(c._children[0]._children)
