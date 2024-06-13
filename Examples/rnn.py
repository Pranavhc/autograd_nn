import numpy as np

from nn.activations import Tanh
from nn.layers import RNN, Dense, Sequential
from nn.losses import MSE
from nn.optim import Adam
from nn.tensor import Tensor
from nn.trainer import Trainer
from nn.utils import DataLoader

X = np.random.randn(1000, 5, 10) # 100 sequences of length 5 with 10 features
Y = np.random.randn(1000, 1) # 100 sequences with 1 target

model = Sequential([
    RNN(10, 5, Tanh()),
    RNN(5, 2, Tanh(), is_last=True),
    Dense(2, 1)
])


optim = Adam(model.get_parameters(), lr=0.5)
criterion = MSE()

datalaoder = DataLoader(X, Y, batch_size=32, shuffle=True, requires_grad=True)

trainer = Trainer(model, criterion, optim)
trainer.train(datalaoder, epochs=10)


# print(model.forward(Tensor(X)).shape)

