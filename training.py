import torch
import torch.nn as nn

# Create model
model = nn.Linear(1, 1)

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Loss function
loss_fn = nn.MSELoss()

# Training data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Training loop
for epoch in range(100):

    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print("Epoch:", epoch + 1, "Loss:", loss.item())

print("Training completed!")