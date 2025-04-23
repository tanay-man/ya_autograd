# mnist_sequential.py
from tensor import Tensor
from nn import Linear, ReLU, Sequential
from cross_entropy import CrossEntropyLoss
from adam import Adam
from mnist import load_mnist
# from data_loader import DataLoader # You won't actually need DataLoader now
import numpy as np # Import numpy

# Load dataset
X_train_np, y_train_np, X_test_np, y_test_np = load_mnist() # X: (N, 784), y: (N,)

# Keep data as NumPy arrays for iteration, convert to Tensors inside the loop/model
# X_train = Tensor(X_train_np, requires_grad=False) # Don't convert whole dataset at once if large
# y_train = Tensor(y_train_np, requires_grad=False)
# X_test = Tensor(X_test_np, requires_grad=False)
# y_test = Tensor(y_test_np, requires_grad=False)

# Define model
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
])

# Define loss and optimizer
loss_fn = CrossEntropyLoss()
# Pass the flattened list of parameters to Adam
optimizer = Adam(model.parameters(), lr=0.001) # Reduced LR is often better for Adam

# Training loop
epochs = 50
num_samples = X_train_np.shape[0]

for epoch in range(epochs):
    total_loss = 0
    # Optional: Shuffle training data each epoch
    permutation = np.random.permutation(num_samples)
    X_train_np_shuffled = X_train_np[permutation]
    y_train_np_shuffled = y_train_np[permutation]

    for i in range(num_samples): # Loop through each sample
        # Get one sample as NumPy array
        X_sample_np = X_train_np_shuffled[i]
        y_sample_np = y_train_np_shuffled[i]

        # Convert the single sample to a Tensor for the forward pass
        # Input data usually doesn't require gradients itself
        X_sample = Tensor(X_sample_np, requires_grad=False)
        # Target label usually doesn't need to be a Tensor for loss calculation,
        # but check your CrossEntropyLoss implementation. Assuming it takes numpy int.
        y_target = y_sample_np # Keep as numpy int

        # Forward pass
        outputs = model(X_sample) # Pass Tensor to model

        # Ensure y_target is in the format expected by loss_fn (e.g., int)
        loss = loss_fn(outputs, y_target)

        # Backward pass
        optimizer.zero_grad() # Zeros grads of parameters tracked by optimizer
        loss.backward()

        # Update parameters - step() updates parameters in-place
        optimizer.step()
        # No need to call model.set_parameters() anymore if Adam updates Tensors in-place
        # updated_params = optimizer.step() # step() returns the list of Tensors
        # model.set_parameters(updated_params) # This is handled by the updated Adam step

        total_loss += loss.data # Accumulate loss data (NumPy value)

        # Optional: Print progress
        if (i + 1) % 10000 == 0:
             print(f"  Epoch {epoch+1}, Sample {i+1}/{num_samples}, Current Avg Loss: {total_loss / (i+1):.4f}")


    avg_loss = total_loss / num_samples
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    # print(model.parameters()) # Printing all parameters can be very verbose

# Evaluation
correct = 0
total = 0
num_test_samples = X_test_np.shape[0]

for i in range(num_test_samples): # Loop through each test sample
    X_sample_np = X_test_np[i]
    y_sample_np = y_test_np[i]

    # Convert sample to Tensor for model
    X_sample = Tensor(X_sample_np, requires_grad=False)
    y_target = y_sample_np # Keep as numpy int

    # Forward pass (no gradient tracking needed for evaluation)
    # You might want a way to disable gradient calculation in your Tensor/model for efficiency
    # e.g., with Tensor.no_grad(): or model.eval() if implemented
    outputs = model(X_sample)

    # Get prediction
    predictions = np.argmax(outputs.data) # Get index of max logit

    # Compare prediction with the actual label
    if predictions == y_target:
        correct += 1
    total += 1 # Increment total for each test sample

# Calculate accuracy
accuracy = correct / total
print(f"Test Accuracy: {accuracy*10:.4f}")