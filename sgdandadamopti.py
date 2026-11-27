import torch
import torch.nn as nn

# Create a simple model
model = nn.Linear(1, 1)

# Create Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Input and correct answer
x = torch.tensor([[3.0]])
actual = torch.tensor([[10.0]])

# Prediction
prediction = model(x)

# Calculate loss
loss = nn.MSELoss()(prediction, actual)

# Calculate gradients
loss.backward()

# Update weights
optimizer.step()

# Clear gradients before the next training step
optimizer.zero_grad()

print("Prediction:", prediction.item())
print("Loss:", loss.item())