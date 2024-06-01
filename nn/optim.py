import numpy as np
from .tensor import Tensor

class Optimizer():
    def __init__(self) -> None:
        self.parameters: list[Tensor] = []
    
    def step(self) -> None:
        raise NotImplementedError('step method must be implemented in the optimizer subclass!')
    
    def zero_grad(self) -> None:
        if len(self.parameters) == 0: raise Exception("No parameters to reset their gradients!")
        
        for param in self.parameters:
            if param.grad: param.grad.data *= 0

class SGD(Optimizer):
    """ ### SGD optimizer with momentum
    
    Update step: 
        ```python 
        v = m * v + (1 - m) * grad
        x = x - learning_rate * v
        ```
    """
    def __init__(self, parameters:list[Tensor], lr:float=0.04, momentum:float=0) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(param.data) for param in self.parameters]

    def step(self) -> None:
        for param, velocity in zip(self.parameters, self.velocity):
            if param.grad is None: raise Exception("Gradient is None!")

            velocity[:] = self.momentum * velocity + (1 - self.momentum) * param.grad.data
            param.data -= self.learning_rate * velocity

class AdaGrad(Optimizer):
    """ ### AdaGrad optimizer
    
    Update step: 
        ```python 
        c = c + grad**2
        x = x - learning_rate * grad / (sqrt(c) + 1e-8)
        ```
    """
    def __init__(self, parameters:list[Tensor], lr:float=0.01, epsilon:float=1e-8) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = lr
        self.epsilon = epsilon
        self.cache = [np.zeros_like(param.data) for param in self.parameters]

    def step(self) -> None:       
        for param, cache in zip(self.parameters, self.cache):
            if param.grad is None: raise Exception("Gradient is None!")

            cache[:] += param.grad.data ** 2
            param.data -= self.learning_rate * param.grad.data / (np.sqrt(cache) + self.epsilon)


class RMSProp(Optimizer):
    """ ### RMSProp optimizer
    
    Update step: 
        ```python 
        c = decay_rate * c + (1 - decay_rate) * grad**2
        x = x - learning_rate * grad / (sqrt(c) + 1e-8)
        ```
    """
    def __init__(self, parameters:list[Tensor], lr:float=0.01, decay_rate:float=0.9, epsilon:float=1e-8) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = lr
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = [np.zeros_like(param.data) for param in self.parameters]

    def step(self):
        for param, cache in zip(self.parameters, self.cache):
            if param.grad is None: raise Exception("Gradient is None!")

            cache[:] = self.decay_rate * cache + (1-self.decay_rate) * param.grad.data**2
            param.data -= self.learning_rate * param.grad.data / (np.sqrt(cache) + self.epsilon)


class Adam(Optimizer):
    """ ### Adam optimizer
    
    Update step:
        ```python 
        momentum = decay_rate_1 * m + (1 - decay_rate_1) * grad
        velocity = decay_rate_2 * v + (1 - decay_rate_2) * grad**2

        m = momentum / (1 - decay_rate_1)
        v = velocity / (1 - decay_rate_2)

        x = x - learning_rate * m / (sqrt(v) + 1e-8)
        ```
    """
    def __init__(self, parameters:list[Tensor], lr:float = 0.01, b1:float=0.9, b2:float=0.995, epsilon:float=1e-8) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = lr
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon

        self.momentum = [np.zeros_like(param.data) for param in parameters]
        self.velocity = [np.zeros_like(param.data) for param in parameters]

    def step(self):
        for param, momentum, velocity in zip(self.parameters, self.momentum, self.velocity):
            if param.grad is None: raise Exception("Gradient is None!")

            momentum[:] = self.b1 * momentum + (1 - self.b1) * param.grad.data
            velocity[:] = self.b2 * velocity + (1 - self.b2) * param.grad.data**2
            
            # since m and v are 0s intially, they remain close to 0s. To avoid this bias, we scale them up.
            m = momentum / (1 - self.b1)
            v = velocity / (1 - self.b2)

            param.data -= self.learning_rate * m / (np.sqrt(v) + self.epsilon)