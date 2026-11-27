import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

x = torch.tensor([[1500.0, 3.0]])

output = model(x)

print("Prediction:", output)