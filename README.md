## Autograd Neural Network

This Neural Network library utilizes 'Automatic Differentiation' (or 'Autograd') to compute gradients of the loss function with respect to the model's parameters. The library is designed to be simple and easy to understand. This library is made for learning purposes.

Look into [tensor.py](https://github.com/Pranavhc/autograd_nn/blob/main/nn/tensor.py) for the autograd system.
Look into [Examples](https://github.com/Pranavhc/autograd_nn/blob/main/Examples/) for usage examples.

```python
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
```

