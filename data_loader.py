import numpy as np
from tensor import Tensor

class DataLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.data.shape[0]
        self.indices = np.arange(self.num_samples)
        
        # Shuffle the data if required
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size  # Total number of batches
    
    def __iter__(self):
        # Iterate over the batches
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            # Get the data for the current batch
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            
            # Convert them to Tensor objects
            X_tensor = Tensor(X_batch, requires_grad=False)
            y_tensor = Tensor(y_batch, requires_grad=False)
            
            yield X_tensor, y_tensor

# Now you can use the custom DataLoader


