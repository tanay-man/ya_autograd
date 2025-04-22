import torch

# Define a simple 2-layer neural network
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with requires_grad=True
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        self.b1 = torch.zeros(hidden_size, requires_grad=True)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True)
        self.b2 = torch.zeros(output_size, requires_grad=True)

    def forward(self, x):
        hidden = torch.relu(x @ self.W1 + self.b1)  # Linear + ReLU
        output = hidden @ self.W2 + self.b2  # Linear
        return output

# Create the network
net = SimpleNN(input_size=2, hidden_size=4, output_size=1)

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# Dummy training data
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=False)
Y = torch.tensor([[3.0], [7.0], [11.0]], requires_grad=False)

# Training loop
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    Y_pred = net.forward(X)
    
    # Compute loss
    loss = mse_loss(Y_pred, Y)
    
    # Backward pass
    loss.backward()
    
    # Update weights manually
    with torch.no_grad():  # Disable autograd during updates
        net.W1 -= learning_rate * net.W1.grad
        net.b1 -= learning_rate * net.b1.grad
        net.W2 -= learning_rate * net.W2.grad
        net.b2 -= learning_rate * net.b2.grad

        # Zero gradients after update
        net.W1.grad.zero_()
        net.b1.grad.zero_()
        net.W2.grad.zero_()
        net.b2.grad.zero_()

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
