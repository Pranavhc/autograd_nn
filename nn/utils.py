import numpy as np
import pickle
from typing import Generator, Union

from .tensor import Tensor

class DataLoader:
    def __init__(self, X:np.ndarray, y:Union[np.ndarray, None]=None, batch_size:int=32, shuffle:bool=False, requires_grad=False) -> None:
        if y is not None and len(X.data) != len(y.data):
            raise ValueError("X and y must have the same length!")
        
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.requires_grad = requires_grad
        self.len = len(X)
        self.n_samples = (self.len + batch_size - 1) // batch_size

    def __call__(self) -> Union[Generator[tuple[Tensor, Tensor], None, None], Generator[Tensor, None, None]]:
        idx = np.arange(self.len)
        if self.shuffle: idx = np.random.permutation(idx)

        for i in range(0, self.len, self.batch_size):
            batch_idx = idx[i:i+self.batch_size]
            if self.y is not None:
                yield Tensor(self.X[batch_idx], requires_grad=self.requires_grad), Tensor(self.y[batch_idx], requires_grad=self.requires_grad)
            else:
                yield Tensor(self.X[batch_idx], requires_grad=self.requires_grad)

def shuffle_dataset(X:np.ndarray, y:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(X.shape[0])         # create an array from 0 to len(X)
    idx = np.random.permutation(idx)    # permute the indexes
    return X[idx], y[idx]               # return the input with permuted indexes

def to_categorical1D(x: np.ndarray, n_col:Union[int, None]=None) -> np.ndarray:
    assert x.ndim == 1, "x should be 1-dimensional"

    if not n_col: n_col = np.max(x) + 1
    return np.eye(n_col)[x] # type: ignore [unnecesary type error]

def save_object(model: object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

