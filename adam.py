# adam.py
from tensor import Tensor
import numpy as np

class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        # parameters is now expected to be a flat list of Tensor objects
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        # Initialize m and v based on the .data attribute of each parameter Tensor
        self.m = [np.zeros_like(p.data) for p in self.parameters if p.requires_grad]
        self.v = [np.zeros_like(p.data) for p in self.parameters if p.requires_grad]
        # Store references to the parameters that actually require gradients
        self.grad_params = [p for p in self.parameters if p.requires_grad]
        self.t = 0

    def step(self):
        self.t += 1
        param_idx = 0 # Index for m and v, which only store moments for grad_params
        for i, p in enumerate(self.parameters): # Iterate through original list
            if not p.requires_grad or p.grad is None: # Skip if no grad needed or available
                continue

            # Ensure grad is a numpy array
            grad_data = p.grad
            if isinstance(grad_data, Tensor): # If grad itself is a Tensor (unlikely but safe)
                 grad_data = grad_data.data

            # Ensure grad_data is not zero-dimensional (scalar) which can cause issues with numpy ops
            if grad_data.ndim == 0:
                grad_data = grad_data.reshape(1) # Reshape scalar grad to 1D array
                # Ensure m and v elements also have compatible shape if initialized from scalar params
                if self.m[param_idx].ndim == 0:
                    self.m[param_idx] = self.m[param_idx].reshape(1)
                if self.v[param_idx].ndim == 0:
                    self.v[param_idx] = self.v[param_idx].reshape(1)


            # m and v updates using the correct index for m and v
            self.m[param_idx] = self.beta1 * self.m[param_idx] + (1 - self.beta1) * grad_data
            self.v[param_idx] = self.beta2 * self.v[param_idx] + (1 - self.beta2) * (grad_data ** 2)

            # Bias correction
            m_hat = self.m[param_idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_idx] / (1 - self.beta2 ** self.t)

            # Parameter update - update p.data in place
            update_value = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Ensure update_value has the same shape as p.data
            if update_value.shape != p.data.shape:
                 # This might happen if bias grad becomes scalar but bias is 1D array
                 if p.data.ndim == 1 and update_value.ndim == 0:
                     update_value = np.full(p.data.shape, update_value.item())
                 # Add more sophisticated shape handling if needed
                 else:
                      raise ValueError(f"Shape mismatch during Adam update: param shape {p.data.shape}, update shape {update_value.shape}")


            p.data -= update_value

            param_idx += 1 # Increment index for m and v only when a grad param is processed

        # Return the original list of parameters (Tensors), which have been updated in-place
        return self.parameters

    def zero_grad(self):
        # Set grad attribute to None for all parameters that require grad
        for p in self.grad_params:
            p.grad = None # Reset gradient attribute