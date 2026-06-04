import numpy as np
from .tensor import Tensor

class Layer:
    """The interface that each layer should implement."""
    def __init__(self) -> None:
        self.params = list()
        self.mode = 'train'

    def get_parameters(self) -> list:
        """Returns the parameters of the layer."""
        return self.params
    
    def eval(self) -> None:
        """Sets the mode of the layer to evaluation mode. Turns off autograd for the paremeters."""
        self.mode = 'eval'
        for param in self.params: param.requires_grad = False

    def train(self) -> None:
        """Sets the mode of the layer to training mode. Turns on autograd for the paremeters."""
        self.mode = 'train'
        for param in self.params: param.requires_grad = True

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
        
        self.weights = Tensor(np.random.uniform(-limit, limit, size=(self.input_size, self.output_size)), requires_grad=True)
        self.bias = Tensor.zeros(self.output_size, requires_grad=True)

        self.params = [self.weights, self.bias]

    def forward(self, input: Tensor) -> Tensor:
        """forward pass. Returns input @ weights + bias"""
        return input.dot(self.weights) + self.bias.expand(0, len(input.data))

# start: Tensor library is incomplete for this to work
class BatchNormalization(Layer):
    """## NOT IMPLEMENTED YET"""
    
    def __init__(self, input_size:int, eps=1e-5, momentum:float=0.1) -> None:
        super().__init__()

        self.input_size = input_size
        self.eps = eps
        self.momentum = momentum

        self.gain = Tensor.ones(self.input_size, requires_grad=True)
        self.bias = Tensor.zeros(self.input_size, requires_grad=True)

        self.running_mean = Tensor.zeros(self.input_size, requires_grad=False)
        self.running_std = Tensor.zeros(self.input_size, requires_grad=False)

        self.params = [self.gain, self.bias]

    def forward(self, input: Tensor) -> Tensor:
        """
        ### Tensor Library is incomplete for now
        ```python
        mean_i = input.mean(axis=0, keepdims=True)
        std_i = input.std(axis=0, keepdims=True)
         
        if self.mode == "train":
            self.running_mean = self.momentum * mean_i + (1 - self.momentum) * self.running_mean
            self.running_std = self.momentum * std_i + (1 - self.momentum) * self.running_std
            input_normalized = (input - mean_i) / (std_i + self.eps)

        else: input_normalized = (input - self.running_mean) / (self.running_std + self.eps)

        out = self.gain * input_normalized + self.bias

        return out
        ```
        """
        raise NotImplementedError("BatchNormalization is not implemented yet.")
# end: Tensor library is incomplete for this to work

# start Tensor indexing is not working well currently
class RNN(Layer):
    def __init__(self, input_size:int, output_size:int, activation:Layer, is_last=False) -> None:
        super().__init__()

        self.input_size = input_size    
        self.output_size = output_size
        self.activation = activation
        self.is_last = is_last

        limit_w = 1 / np.sqrt(self.input_size)
        limit_rw = 1 / np.sqrt(self.input_size)

        self.w_i = Tensor(np.random.uniform(-limit_w, limit_w, (self.input_size, self.output_size)), requires_grad=True)
        self.w_rw = Tensor(np.random.uniform(-limit_rw, limit_rw, (self.output_size, self.output_size)), requires_grad=True)
        self.bias = Tensor(np.zeros(self.output_size), requires_grad=True)

        self.params = [self.w_i, self.w_rw, self.bias]

    def forward(self, input: Tensor) -> Tensor:
        """forward pass. Returns the output of the RNN."""
        batch_size, timesteps, n_features = input.shape

        self.prev = Tensor.zeros((batch_size, self.output_size), requires_grad=True)
        self.output = Tensor.zeros((batch_size, timesteps, self.output_size), requires_grad=True)
        
        self.successive_states = Tensor.zeros((batch_size, timesteps+1, self.output_size), requires_grad=True)

        for t in range(timesteps):
            self.output[:, t] = self.activation.forward(input[:, t].dot(self.w_i) + self.prev.dot(self.w_rw) + self.bias)
            self.prev = self.output[:, t]
            self.successive_states[:, t+1] = self.output[:, t]
        
        if self.is_last: return self.successive_states[:, -1]
        return self.successive_states[:, 1:]
# end: Tensor indexing is not working well currently

class Dropout(Layer):
    """A dropout layer."""
    def __init__(self, drop_rate:float=0.3) -> None:
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, input:Tensor) -> Tensor:
        """forward pass. Returns input with some elements zeroed out."""
        if self.mode=='train': self.mask = Tensor(np.random.choice([0, 1], size=input.shape, p=[self.drop_rate, 1-self.drop_rate]), requires_grad=True)
        else: self.mask = Tensor(np.ones_like(input.data) * (1 - self.drop_rate), requires_grad=False)
        return input * self.mask