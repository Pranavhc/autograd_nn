from typing import Union
import numpy as np
from tqdm import tqdm

from .tensor import Tensor
from .layers import Layer
from .losses import Loss
from .optim import Optimizer
from .utils import DataLoader

class Trainer:
    def __init__(self, model: Layer, criterion: Loss, optim: Optimizer) -> None:
        self.model = model
        self.criterion = criterion
        self.optim = optim

        self.error = {'train': [], 'val': []}
        self.accuracy = {'train': [], 'val': []}

    def _train_on_batch(self, X:Tensor, y:Tensor) -> tuple[float, np.ndarray]:
        self.model.train()
        self.optim.zero_grad()

        pred = self.model.forward(X)
        loss = self.criterion(y, pred)
        loss.backward()
        self.optim.step()

        return (float(loss.data.mean()), pred.data)
    
    def _eval_on_batch(self, X:Tensor, y:Tensor) -> tuple[float, np.ndarray]:
        self.model.eval()
        pred = self.model.forward(X)
        loss = self.criterion(y, pred)
        return (float(loss.data.mean()), pred.data)
    
    def acc(self, y_pred: np.ndarray, y:np.ndarray) -> float:
        assert y_pred.shape == y.shape, "Shapes of y_pred and y must be the same"

        if y_pred.ndim >= 1 and y.ndim >= 1:
            return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / len(y) * 100
        else: return np.sum(y_pred == y) / len(y) * 100

    def train(self, train_data:DataLoader, val_data:Union[DataLoader, None]=None, epochs:int=10, show_accuracy:bool=False, verbose:bool=True) -> Union[tuple[dict, dict], dict]:
        for e in range(epochs):
            train_batch_loss, val_batch_loss = [], []
            train_batch_acc, val_batch_acc = [], []

            with tqdm(train_data(), unit='batch', disable=not verbose) as pbar:
                for X_t, y_t in pbar:
                    pbar.set_description(f"Epoch {e+1}/{epochs}")
                    pbar.total = train_data.n_samples
                    pbar.bar_format = "{l_bar}{bar:20}| {n_fmt}/{total_fmt}{postfix}"

                    train_loss, pred_y_t = self._train_on_batch(X_t, y_t)
                    train_batch_loss.append(train_loss)

                    if show_accuracy: 
                        train_batch_acc.append(self.acc(pred_y_t, y_t.data))
                        pbar.set_postfix(accuracy=f"{np.mean(train_batch_acc):.4f}", loss=f"{np.mean(train_batch_loss):.4f}")
                    else: pbar.set_postfix(loss=f"{np.mean(train_batch_loss):.4f}")

                self.error['train'].append(float(np.mean(train_batch_loss)))
                if show_accuracy: self.accuracy['train'].append(np.mean(train_batch_acc))

            if val_data is not None:
                for X_v, y_v in val_data():
                    val_loss, pred_y_v = self._eval_on_batch(X_v, y_v)
                    val_batch_loss.append(val_loss)

                    if show_accuracy: val_batch_acc.append(self.acc(pred_y_v, y_v.data))

                if show_accuracy: tqdm.write(f"val_accuracy={np.mean(val_batch_acc):.4f}, val_loss={np.mean(val_batch_loss):.4f} \n")
                else: tqdm.write(f"val_loss={np.mean(val_batch_loss):.4f} \n")

                self.error['val'].append(float(np.mean(val_batch_loss)))
                if show_accuracy: self.accuracy['val'].append(np.mean(val_batch_acc))    
    
        if show_accuracy: return self.error, self.accuracy
        return self.error