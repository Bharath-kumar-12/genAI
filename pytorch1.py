import torch
import torch.nn as nn

# Create a neural network
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

# Input data
x = torch.tensor([[2.0, 3.0]])

# Prediction
output = model(x)

print("Input:", x)
print("Prediction:", output)