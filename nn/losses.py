from .tensor import Tensor

class Loss:
    """All loss functions must implement this interface."""
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Compute the loss."""
        raise NotImplementedError

class MSE(Loss):
    """Mean Squared Error Loss Function."""
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return ((y_true - y_pred) * (y_true - y_pred)).mean()

class BinaryCrossEntropy(Loss):
    """Binary CrossEntropy Loss Function."""
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return -(y_true * (y_pred + 1e-7).log() + (1 - y_true) * (1 - y_pred + 1e-7).log()).mean()
        
class CategoricalCrossEntropy(Loss):
    """Categorical CrossEntropy Loss Function."""
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return -(y_true * (y_pred + 1e-7).log()).mean()
    
