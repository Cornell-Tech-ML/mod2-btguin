"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5. START

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        # Initialize three linear layers
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        x = x.contiguous()
        # Ensure x has the correct last dimension
        expected_in_size = self.layer1.weights.value.shape[0]
        if x.shape[-1] != expected_in_size:
            raise ValueError(f"Expected input with last dimension {expected_in_size}, got {x.shape[-1]}")
        # Forward pass through the network with ReLU and Sigmoid activations
        out = self.layer1.forward(x).relu()
        out = self.layer2.forward(out).relu()
        out = self.layer3.forward(out).sigmoid()
        return out  # Shape: (N, 1)

# Define Linear and Network classes before TensorTrain
class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # Initialize weights and bias using RParam
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)

    def forward(self, x):
        x = x.contiguous()
        in_size = self.weights.value.shape[0]
        if x.shape[-1] != in_size:
            raise ValueError(f"Expected input with last dimension {in_size}, got {x.shape[-1]}")
        out_size = self.weights.value.shape[1]

        x_shape = x.shape[:-1]  # All dimensions except the last (input features)
        batch_size = int(minitorch.operators.prod(x_shape)) if x_shape else 1

        # Reshape x to (batch_size, in_size)
        x = x.view(batch_size, in_size)  # Shape: (batch_size, in_size)
        # Expand x and weights for broadcasting
        x_expanded = x.view(batch_size, in_size, 1)  # Shape: (batch_size, in_size, 1)
        weights_expanded = self.weights.value.view(1, in_size, out_size)  # Shape: (1, in_size, out_size)

        # Perform element-wise multiplication and sum over in_size dimension
        x_w = x_expanded * weights_expanded  # Shape: (batch_size, in_size, out_size)
        z = x_w.sum(1) + self.bias.value  # Sum over in_size dimension, shape: (batch_size, out_size)

        # Reshape z back to match input batch dimensions
        z = z.view(*x_shape, out_size)  # Shape: (..., out_size)

        return z


# TODO: Implement for Task 2.5. END



def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
