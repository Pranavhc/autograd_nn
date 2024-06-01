from .tensor import Tensor
from .layers import Layer

class ReLu(Layer):
  """ Rectified Linear Unit (ReLU) activation function. """
  def forward(self, input: Tensor) -> Tensor:
    return input.relu()

class Tanh(Layer):
  """ Hyperbolic Tangent (Tanh) activation function. """
  def forward(self, input: Tensor) -> Tensor:
    return input.tanh()

class Sigmoid(Layer):
  """ Sigmoid activation function. """
  def forward(self, input: Tensor) -> Tensor:
    return input.sigmoid()  

class Softmax(Layer):
  """ Softmax activation function. """
  def forward(self, input: Tensor) -> Tensor:
    return input.softmax()