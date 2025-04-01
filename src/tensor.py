import numpy as np
from typing import Any

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
        return f"Data = {self.data}\nrequires_grad = {self.requires_grad}\ngrad = {self.grad}\ngrad_func = {self._grad_function}\n"
    
    def backward(self):
        pass 