# nn.py
from tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        # Initialize weights with He initialization
        self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        # Ensure input is a Tensor if it's not already
        if not isinstance(x, Tensor):
             x = Tensor(x, requires_grad=False) # Assume input data doesn't need grad tracking itself
        self.input = x
        return x.matmul(self.weight) + self.bias

    def parameters(self):
        # Returns the list of parameters for this layer
        return [self.weight, self.bias]

    def set_parameters(self, weight_data, bias_data):
        """Set the parameters of the Linear layer using NumPy arrays."""
        # Ensure the data being set matches the expected shapes
        assert weight_data.shape == self.weight.data.shape, f"Weight shape mismatch: expected {self.weight.data.shape}, got {weight_data.shape}"
        assert bias_data.shape == self.bias.data.shape, f"Bias shape mismatch: expected {self.bias.data.shape}, got {bias_data.shape}"

        self.weight.data = weight_data
        self.bias.data = bias_data
        # Re-enable gradients if they were turned off during update
        self.weight.requires_grad = True
        self.bias.requires_grad = True


class ReLU:
    def __call__(self, x):
        self.input = x
        return x.relu()

    def parameters(self):
        # ReLU has no trainable parameters
        return []

    def set_parameters(self, *args, **kwargs):
        """ReLU has no parameters to set."""
        pass


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        # Ensure input is a Tensor
        if not isinstance(x, Tensor):
             x = Tensor(x, requires_grad=False)
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Return a flat list of all parameters in the sequential model."""
        params = []
        for layer in self.layers:
            # Use extend to add all parameters from the layer's list
            params.extend(layer.parameters())
        return params

    def set_parameters(self, flat_params_list):
        """
        Set parameters for each layer using a flat list of parameter Tensors.
        `flat_params_list` should be a flat list containing all parameter Tensors
        in the same order as returned by parameters().
        """
        param_idx = 0
        for layer in self.layers:
            layer_params = layer.parameters() # Get the expected parameter structure for this layer
            num_layer_params = len(layer_params)

            if num_layer_params > 0:
                # Collect the required parameters for this layer from the flat list
                current_layer_params_data = []
                for i in range(num_layer_params):
                    # Adam updates .data in-place, so we extract the updated .data
                    current_layer_params_data.append(flat_params_list[param_idx + i].data)

                # Use the layer's specific set_parameters method
                # Linear expects (weight_data, bias_data)
                if isinstance(layer, Linear):
                     layer.set_parameters(current_layer_params_data[0], current_layer_params_data[1])
                # Add other layer types here if they have parameters and a set_parameters method
                # else:
                #    layer.set_parameters(*current_layer_params_data) # Generic attempt

                # Move the index forward by the number of parameters consumed
                param_idx += num_layer_params
            # else: layer has no parameters (like ReLU), do nothing