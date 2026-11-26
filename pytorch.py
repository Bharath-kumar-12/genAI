
import torch
import numpy as np

print("Welcome to PyTorch!")

print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)

x = torch.rand(3, 3)

print("\nMy First Tensor:")
print(x)

print("\nCUDA available:", torch.cuda.is_available())
