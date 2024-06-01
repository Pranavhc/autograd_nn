## Autograd Neural Network

This Neural Network library utilizes 'Automatic Differentiation' (or 'Autograd') to compute gradients of the loss function with respect to the model's parameters. The library is designed to be simple and easy to understand. This library is made for learning purposes.

**The main components of the library are**:

1. **Tensor**: A multi-dimensional array with support for autograd.
2. **Layers**: A collection of layers that can be stacked together to form a neural network.
3. **Activations**: A collection of activation functions.
4. **Optimizers**: A collection of optimization algorithms.
5. **Losses**: A collection of loss functions that can be used to evaluate the performance.
6. **Utils**: Contains utility functions like DataLoader, save_object, load_object, etc.
7. **Trainer**: A class used to train a neural network. Or you can write a custom training loop.

```python
from nn.layers import Sequential, Dense, Dropout
from nn.activations import ReLu, Softmax
from nn.optim import Adam
from nn.losses import CategoricalCrossEntropy as CCE
from nn.utils import DataLoader, save_object, load_object
from nn.trainer import Trainer

clf = Sequential([
    Dense(28*28, 128), 
    ReLu(),
    Dropout(0.3),

    Dense(128, 64), 
    ReLu(),
    Dropout(0.3),

    Dense(64, 10),
    Softmax()
])

train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True, autograd=True)
val_loader = DataLoader(X_val, y_val, batch_size=128, shuffle=True, autograd=False)

trainer = Trainer(clf, CCE(), Adam(clf.get_parameters(), lr=0.01))
history = trainer.train(train_loader, val_loader, epochs=10)
```

