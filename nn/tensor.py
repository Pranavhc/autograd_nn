from typing import Union
import numpy as np

class Tensor():
    def __init__(self, data, requires_grad:bool=False, parents=None, op:Union[str, None]=None, id:Union[int, None]=None) -> None:
        self.data = np.array(data)
        self.op = op
        self.parents = parents
        self.grad = None
        self.requires_grad = requires_grad
        self.id = id if id is not None else np.random.randint(0, 100000)
        self.children = {}
        self.shape = self.data.shape

        # track the no. of children a tensor has
        if(parents): 
            for parent in parents: 
                parent.children[self.id] = parent.children.get(self.id, 0) + 1

    @classmethod
    def zeros(cls, arr_or_shape: Union[np.ndarray, tuple, int], requires_grad:bool=False):
        if not isinstance(arr_or_shape, (np.ndarray, tuple, int)):
            raise Exception("You must provide either an array or shape!")
        
        if isinstance(arr_or_shape, np.ndarray): return cls(np.zeros_like(arr_or_shape), requires_grad=requires_grad)
        elif isinstance(arr_or_shape, (tuple, int)): return cls(np.zeros(arr_or_shape), requires_grad=requires_grad)
    
    @classmethod
    def ones(cls, arr_or_shape: Union[np.ndarray, tuple, int], requires_grad:bool=False):
        if not isinstance(arr_or_shape, (np.ndarray, tuple, int)):
            raise Exception("You must provide either an array or shape!")
        
        if isinstance(arr_or_shape, np.ndarray): return cls(np.ones_like(arr_or_shape), requires_grad=requires_grad)
        elif isinstance(arr_or_shape, (tuple, int)): return cls(np.ones(arr_or_shape), requires_grad=requires_grad)

    def to_numpy(self) -> np.ndarray: return self.data

    def received_grads_from_all_children(self) -> bool:
        for id, count in self.children.items():
            if count != 0: return False
        return True
    
    def backward(self, grad=None, grad_origin=None) -> None:       
        if not self.requires_grad: raise Exception("Cannot backpropagate through a non-autograd tensor!") 

        if grad is None: grad = Tensor(np.ones_like(self.data))
        if grad_origin:
            if self.children[grad_origin.id] == 0: raise Exception("Cannot backpropogate more than once!")
            else: self.children[grad_origin.id] -= 1 # can backpropogate at least once

        # accumulates gradients from several children
        if self.grad is None: self.grad = grad
        else: self.grad += grad


        if self.parents and (self.received_grads_from_all_children() or grad_origin is None):
            # backward methods for each operation

            # start: index (__getitem__) and __setitem__ don't work as I would like them to
            if self.op == "index":
                new_grad = np.zeros_like(self.parents[0].data)
                # Assign the gradient values directly using advanced indexing
                new_grad[self.idx] += self.grad  # type: ignore
                # Backpropagate the gradient to the parent tensor
                self.parents[0].backward(Tensor(new_grad), self)

            if self.op == 'setitem':
                np.add.at(self.parents[0].grad, self.idx, self.grad.data) # type: ignore
                self.parents[0].backward(Tensor(self.parents[0].grad), self)

                if len(self.parents) > 1:
                    self.parents[1].backward(self.grad[self.idx], self) # type: ignore
            # end: index (__getitem__) and __setitem__ 

            if self.op == 'add':
                self.parents[0].backward(self.grad, self)
                if len(self.parents) > 1: self.parents[1].backward(self.grad, self)

            if self.op == "neg":
                self.parents[0].backward(self.grad.__neg__(), self)

            if self.op == "sub":
                self.parents[0].backward(Tensor(self.grad.data), self)
                if len(self.parents) > 1: self.parents[1].backward(Tensor(self.grad.__neg__().data), self)

            if self.op == "rsub":
                self.parents[0].backward(self.grad.__neg__(), self)

            if self.op == "mul":
                self.parents[0].backward(self.grad * self.parents[1], self)
                self.parents[1].backward(self.grad * self.parents[0], self)          

            if self.op and "sum" in self.op:
                dim = int(self.op.split("_")[1])
                self.parents[0].backward(self.grad.expand(dim, self.parents[0].data.shape[dim]), self)

            if self.op and "expand" in self.op:
                dim = int(self.op.split("_")[1])
                self.parents[0].backward(self.grad.sum(dim), self)
            
            if self.op == "transpose":
                self.parents[0].backward(self.grad.transpose(), self)
            
            if self.op == "dot":
                p0 = self.parents[0]
                p1 = self.parents[1]
                
                p0.backward(self.grad.dot(p1.transpose()), self)
                p1.backward(self.grad.transpose().dot(p0).transpose(), self)
                
            if self.op == "log":
                self.parents[0].backward(Tensor(self.grad.data * (1 / self.parents[0].data)), self)

            if self.op == "mean":
                self.parents[0].backward(Tensor(self.grad.data / np.prod(self.parents[0].data.shape)), self)

            if self.op == "exp":
                self.parents[0].backward(Tensor(self.grad.data * np.exp(self.data)), self)

            if self.op == "relu":
                self.parents[0].backward(Tensor(np.where(self.parents[0].data < 0, 0, 1) * self.grad.data), self)

            if self.op == "tanh":
                # tanh_val = np.tanh(self.parents[0].data)
                tanh = self.data
                self.parents[0].backward(Tensor((1 - tanh ** 2) * self.grad.data), self)
            
            if self.op == "sigmoid":
                sigmoid = self.data
                self.parents[0].backward(Tensor(self.grad.data * sigmoid * (1 - sigmoid)), self)

            if self.op == "softmax":
                softmax = self.data
                grad = np.multiply(self.grad.data, softmax)
                grad -= np.sum(grad, axis=-1, keepdims=True) * softmax

                self.parents[0].backward(Tensor(grad), self)

    # start: __getitem__ and __setitem__ currently don't work as I would like them to
    def __getitem__(self, idx):
        if self.requires_grad:
            new = Tensor(self.data[idx], requires_grad=True, parents=[self], op="index")
            new.idx = idx # type: ignore
            return new
        return Tensor(self.data[idx])
    
    def __setitem__(self, idx, value):
        self.data[idx] = value.data if isinstance(value, Tensor) else value

        # Track the operation for autograd
        if self.requires_grad:
            if isinstance(value, Tensor):
                new = Tensor(self.data, requires_grad=True, parents=[self, value], op="setitem")
            else:
                new = Tensor(self.data, requires_grad=True, parents=[self], op="setitem")
            new.idx = idx  # type: ignore
            self = new
    # end: __getitem__ and __setitem__
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            if self.requires_grad and other.requires_grad:
                return Tensor(self.data + other.data, requires_grad=True, parents=[self, other], op='add')
            return Tensor(self.data + other.data)
        elif isinstance(other, (int, float)):
            if self.requires_grad:
                return Tensor(self.data + other, requires_grad=True, parents=[self], op='add')
            return Tensor(self.data + other)
        else: raise TypeError("other must be a Tensor or a number!")
    
    def __neg__(self):
        if self.requires_grad: return Tensor(self.data * -1, requires_grad=True, parents=[self], op="neg")
        return Tensor(self.data * -1)
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            if(self.requires_grad and other.requires_grad):
                return Tensor(self.data - other.data, requires_grad=True, parents=[self,other], op="sub")
            return Tensor(self.data - other.data)
        elif isinstance(other, (int, float)):
            if self.requires_grad:
                return Tensor(self.data - other, requires_grad=True, parents=[self], op="sub")
            return Tensor(self.data - other)
        else: raise TypeError("other must be a Tensor or a number!")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            if self.requires_grad:
                return Tensor(other - self.data, requires_grad=True, parents=[self], op="rsub")
            return Tensor(other - self.data)
        else: raise TypeError("other must be a number!")
    
    def __mul__(self, other):
        if(self.requires_grad and other.requires_grad):
            return Tensor(self.data * other.data, requires_grad=True, parents=[self,other], op="mul")
        return Tensor(self.data * other.data)
    
    def sum(self, dim:int):
        if(self.requires_grad):
            return Tensor(self.data.sum(dim), requires_grad=True, parents=[self], op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    
    def expand(self, dim:int, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)
        
        if(self.requires_grad):
            return Tensor(new_data, requires_grad=True, parents=[self], op="expand_"+str(dim))
        return Tensor(new_data)
    
    def transpose(self):
        if(self.requires_grad):
            return Tensor(self.data.transpose(), requires_grad=True, parents=[self], op="transpose")
        return Tensor(self.data.transpose())
    
    def dot(self, x):
        if(self.requires_grad):
            return Tensor(self.data.dot(x.data), requires_grad=True, parents=[self,x], op="dot")
        return Tensor(self.data.dot(x.data))
    
    def log(self):
        if(self.requires_grad):
            return Tensor(np.log(self.data), requires_grad=True, parents=[self], op="log")
        return Tensor(np.log(self.data))
    
    def mean(self):
        if self.requires_grad:
            return Tensor(np.mean(self.data), requires_grad=True, parents=[self], op="mean")
        return Tensor(np.mean(self.data))
    
    def exp(self):
        if self.requires_grad:
            return Tensor(np.exp(self.data), requires_grad=True, parents=[self], op="exp")
        return Tensor(np.exp(self.data))
        
    def relu(self):
        if self.requires_grad:
            return Tensor(np.maximum(0, self.data), requires_grad=True, parents=[self], op="relu")
        return Tensor(np.maximum(0, self.data))
    
    def tanh(self):
        if self.requires_grad:
            return Tensor(np.tanh(self.data), requires_grad=True, parents=[self], op="tanh")
        return Tensor(np.tanh(self.data))

    def sigmoid(self):
        data = np.clip(self.data, -709, 709)  # np.exp(709) is the largest representable number

        if self.requires_grad:
            return Tensor(1 / (1 + np.exp(-data)), requires_grad=True, parents=[self], op="sigmoid")
        return Tensor(1 / (1 + np.exp(-data)))
    
    def softmax(self):
        e_x = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        softmax = e_x / np.sum(e_x, axis=-1, keepdims=True)
        if self.requires_grad:
            return Tensor(softmax, requires_grad=True, parents=[self], op="softmax")
        return Tensor(softmax)

    def __repr__(self) -> str:
        return str(self.data.__repr__())
    
    def __str__(self) -> str:
        return str(self.data.__str__()) 