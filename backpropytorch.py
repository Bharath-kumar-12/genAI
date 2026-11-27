import torch

# Weight
weight = torch.tensor(2.0, requires_grad=True)

# Input
x = torch.tensor(3.0)

# Prediction
prediction = weight * x

# Correct answer
actual = torch.tensor(10.0)

# Calculate loss
loss = (prediction - actual) ** 2

# Backpropagation
loss.backward()

print("Prediction:", prediction.item())
print("Loss:", loss.item())
print("Gradient:", weight.grad.item())