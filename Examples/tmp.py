from nn.tensor import Tensor
import numpy as np

x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
w = Tensor(np.array([4.0, -1.0]), requires_grad=True)

# Some operation
out = (x * w).sum(dim=0)

# Backpropogation
out.backward()

print("x.grad:", x.grad)  # [4.0, -1.0]
print("w.grad:", w.grad)  # [2.0, 3.0]