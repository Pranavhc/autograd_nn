import numpy as np
from .tensor import Tensor

class Layer:
    """The interface that each layer should implement."""
    def __init__(self) -> None:
        self.params = list()
        self.mode = 'train'

    def get_parameters(self) -> list:
        return self.params
    
    def eval(self) -> None:
        self.mode = 'eval'
        for param in self.params: param.autograd = False

    def train(self) -> None:
        self.mode = 'train'
        for param in self.params: param.autograd = True

    def forward(self, input: Tensor) -> Tensor:
        """forward pass. Returns the output of the layer."""
        raise NotImplementedError   
    
class Sequential(Layer):
    def __init__(self, layers: list[Layer]) -> None:
        super().__init__()

        self.layers = layers
        self.params = [param for layer in self.layers for param in layer.get_parameters()]

    def eval(self) -> None:
        for layer in self.layers: layer.eval()

    def train(self) -> None:
        for layer in self.layers: layer.train()
    
    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
        self.params += layer.get_parameters()

    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer.forward(input)
        return input

class Dense(Layer):
    """A dense (fully connected) layer in a neural network."""
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        limit = 1/np.sqrt(self.input_size)
        
        self.weights = Tensor(np.random.uniform(-limit, limit, size=(self.input_size, self.output_size)), autograd=True)
        self.bias = Tensor.zeros(self.output_size, autograd=True)

        self.params = [self.weights, self.bias]

    def forward(self, input: Tensor) -> Tensor:
        """forward pass. Returns input @ weights + bias"""
        return input.dot(self.weights) + self.bias.expand(0, len(input.data))
    
class Dropout(Layer):
    """A dropout layer."""
    def __init__(self, drop_rate:float=0.3) -> None:
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, input:Tensor) -> Tensor:
        """forward pass. Returns input with some elements zeroed out."""
        if self.mode=='train': self.mask = Tensor(np.random.choice([0, 1], size=input.shape, p=[self.drop_rate, 1-self.drop_rate]), autograd=True)
        else: self.mask = Tensor(np.ones_like(input.data) * (1 - self.drop_rate), autograd=False)
        return input * self.mask