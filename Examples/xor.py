import numpy as np
import matplotlib.pyplot as plt

from nn.activations import Tanh
from nn.layers import Sequential, Dense
from nn.losses import MSE
from nn.optim import Adam
from nn.tensor import Tensor
from nn.utils import DataLoader, save_object, load_object

X_train = np.reshape([[0,0], [0, 1], [1, 0], [1, 1]], (4, 2))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1))

data = DataLoader(X_train, y_train, batch_size=4, shuffle=True, requires_grad=True)

model = Sequential([
    Dense(2, 24),
    Tanh(),
    Dense(24, 4),
    Tanh(),
    Dense(4, 1),
])

criterion = MSE()
optim = Adam(model.get_parameters(), lr=0.01)

# custom training loop
model.train()
for e in range(220):
    for X, y in data():
        optim.zero_grad()
        pred = model.forward(X)
        loss = criterion(y, pred)
        loss.backward()
        optim.step()

    print(f"Epoch: {e} - Loss: {loss.data.sum()}")

save_object(model, "Examples/models/xor.model")

# Inference
loaded_model = load_object("Examples/models/xor.model")
loaded_model.eval()

# Decision Boundary
def decision_boundary():
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            z = loaded_model.forward(Tensor(np.array([[x, y]]))).to_numpy()[0, 0]
            points.append([x, y, z])
    points = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
decision_boundary()

# Predictions
print("\nInference:")
pred:Tensor = loaded_model.forward(Tensor(X_train, requires_grad=False))
for y_hat, y_train in zip(pred.data[:,0], y_train[:,0]): print(f"Prediction: {y_hat:.4f} - Actual: {y_train}")