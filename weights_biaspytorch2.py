import torch

# Input
x = torch.tensor(5.0)

# Weight
weight = torch.tensor(2.0)

# Bias
bias = torch.tensor(3.0)

# Prediction
y = (x * weight) + bias

print("Prediction:", y.item())