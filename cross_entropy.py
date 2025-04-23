# cross_entropy.py
from tensor import Tensor
import numpy as np

class CrossEntropyLoss:
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        logits: Tensor of shape (num_classes,)
        targets: One-hot encoded Tensor of shape (num_classes,)
        """
        # Apply softmax
        exp_logits = np.exp(logits.data - np.max(logits.data))  # for numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Cross-entropy loss: Find the class with the highest probability in one-hot encoded targets
        target_index = np.argmax(targets.data)  # Get the index of the class with label 1
        log_likelihood = -np.log(probs[target_index])  # Indexing with the correct class
        loss = log_likelihood

        # Store softmax probs for backward
        self.probs = probs
        self.target = target_index  # Store the index of the target class
        return Tensor(loss, requires_grad=True, children=[logits], grad_function="CrossEntropy")

    def backward(self, logits: Tensor):
        """
        Custom backward function to compute gradient of loss w.r.t logits.
        """
        grad = self.probs
        grad[self.target] -= 1
        logits.backward(grad)
