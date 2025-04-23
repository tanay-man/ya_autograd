import numpy as np
from typing import Optional, Union

class Tensor:
    def __init__(self,
                 data: Union[np.ndarray, float, int],
                 requires_grad: bool = False,
                 grad: np.ndarray = None,
                 children: list["Tensor"] = None,
                 grad_function=None):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.requires_grad = requires_grad
        self.grad = grad
        self._children = children or []
        self._grad_function = grad_function

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad}, grad_func={self._grad_function})"

    def __getitem__(self, index):
        return Tensor(self.data[index], self.requires_grad, self.grad, self._children, self._grad_function)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data + other.data,
                      self.requires_grad or other.requires_grad,
                      None, [self, other], "Add")

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data - other.data,
                      self.requires_grad or other.requires_grad,
                      None, [self, other], "Sub")

    def __neg__(self):
        return Tensor(-self.data,
                      self.requires_grad,
                      None, [self], "Neg")

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data * other.data,
                      self.requires_grad or other.requires_grad,
                      None, [self, other], "Mult")

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data / other.data,
                      self.requires_grad or other.requires_grad,
                      None, [self, other], "Div")

    def __pow__(self, power):
        return Tensor(self.data ** power,
                      self.requires_grad,
                      None, [self], "Pow")

    def max(self):
        return Tensor(np.max(self.data), self.requires_grad, None, [self], "Max")

    def matmul(self, other):
        if self.data.ndim < 1 or other.data.ndim < 1:
            raise ValueError(f"matmul: Both operands must have at least 1 dimension. Got {self.data.ndim} and {other.data.ndim}.")
        return Tensor(np.matmul(self.data, other.data),
                    self.requires_grad or other.requires_grad,
                    None, [self, other], "MatMul")
    def T(self):
        return Tensor(self.data.T, self.requires_grad, None, [self], "Transpose")

    def mean(self):
        return Tensor(np.mean(self.data), self.requires_grad, None, [self], "Mean")

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data > other.data, False)

    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data < other.data, False)

    # Activation functions
    def relu(self):
        return Tensor(np.maximum(0, self.data), self.requires_grad, None, [self], "ReLU")

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        return Tensor(sig, self.requires_grad, None, [self], "Sigmoid")

    def tanh(self):
        return Tensor(np.tanh(self.data), self.requires_grad, None, [self], "Tanh")

    def backward(self, grad: Optional[np.ndarray] = None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self._grad_function == "Add":
            for child in self._children:
                if child.requires_grad:
                    child.backward(grad)

        elif self._grad_function == "Sub":
            self._children[0].backward(grad)
            self._children[1].backward(-grad)

        elif self._grad_function == "Neg":
            self._children[0].backward(-grad)

        elif self._grad_function == "Mult":
            left, right = self._children
            if left.requires_grad:
                left.backward(grad * right.data)
            if right.requires_grad:
                right.backward(grad * left.data)

        elif self._grad_function == "Div":
            left, right = self._children
            if left.requires_grad:
                left.backward(grad / right.data)
            if right.requires_grad:
                right.backward(-grad * left.data / (right.data ** 2))

        elif self._grad_function == "Pow":
            base = self._children[0]
            if base.requires_grad:
                # derivative of x^n is n * x^(n-1)
                base.backward(grad * (self.data ** (1 - 1e-8)))

        elif self._grad_function == "Mean":
            child = self._children[0]
            grad_input = grad * np.ones_like(child.data) / child.data.size
            child.backward(grad_input)

        elif self._grad_function == "MatMul":
            A, B = self._children
            if A.requires_grad:
                A.backward(np.matmul(grad, B.data.T))
            if B.requires_grad:
                B.backward(np.matmul(A.data.T, grad))

        elif self._grad_function == "Transpose":
            self._children[0].backward(grad.T)

        elif self._grad_function == "ReLU":
            child = self._children[0]
            relu_grad = grad * (child.data > 0)
            child.backward(relu_grad)

        elif self._grad_function == "Sigmoid":
            sigmoid_output = 1 / (1 + np.exp(-self._children[0].data))
            child = self._children[0]
            child.backward(grad * sigmoid_output * (1 - sigmoid_output))

        elif self._grad_function == "Tanh":
            tanh_output = np.tanh(self._children[0].data)
            child = self._children[0]
            child.backward(grad * (1 - tanh_output ** 2))

        elif self._grad_function == "Max":
            child = self._children[0]
            mask = (child.data == np.max(child.data))
            child.backward(grad * mask.astype(float))
