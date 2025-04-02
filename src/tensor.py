import numpy as np
from typing import Optional

class Tensor:
    def __init__(self,
                 data: np.ndarray,
                 requires_grad:bool = False,
                 grad: np.ndarray = None,
                 children: list["Tensor"] = None,
                 grad_function = None):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = grad
        self._children = children
        self._grad_function = grad_function

    def __mul__(self, other: "Tensor") -> "Tensor":
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(np.multiply(self.data, other.data), requires_grad, None, [self, other], "Mult")
    
    def __add__(self, other: "Tensor") -> "Tensor":
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(np.add(self.data, other.data), requires_grad, None, [self, other], "Add")

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad}, grad_func={self._grad_function})"

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

        elif self._grad_function == "Mult":
            left, right = self._children
            if left.requires_grad:
                left.backward(grad * right.data)  
            if right.requires_grad:
                right.backward(grad * left.data) 
        