import torch
from torch.autograd import Function


class Exp(Function):
    """
        We can implement our own custom autograd Functions by subclassing
        torch.autograd.Function and implementing the forward and backward passes
        which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, i):
        """
            In the forward pass we receive a Tensor containing the input and return
            a Tensor containing the output. ctx is a context object that can be used
            to stash information for backward computation. You can cache arbitrary
            objects for use in the backward pass using the ctx.save_for_backward method.
        """
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
        """
        result, = ctx.saved_tensors
        return grad_output * result


class Softmax(Function):
    """ simple softmax implementation with torch autograd """
    
    @staticmethod
    def forward(ctx, i):
        """
            softmax = e^x_i / sum_i(e^x)
        """
        result = i.exp() / i.exp().sum()
        ctx.save_for_backward(result)
        return result
    
    def backward(ctx, grad_output):
        """
            derivative = res * (1 - res)
        """
        result, = ctx.saved_tensors
        return grad_output * result * (1 - result)


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None # function 
        self._prev = set(_children) # set of Value objects
        # not sure why do we need this one
        self._op = _op # the op that produced this node, string ('+', '-', ....)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data + self.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data * self.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), '**')

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))
        out._backward = _backward

        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), 'relu')

        def _backward():
            self.grad += out.grad if self.data > 0 else 0
        out._backward = _backward

        return out
    
    def exp(self):
        out = Value(self.data.exp(), (self,), 'exp')
        
        def _backward():
            self.grad += out.grad * self.data.exp()
        out._backward = _backward
        
        return out
    
    def sum(self, axes=0):
        out = Value(self.data.sum(axes), (self,), 'sum')
        
        # Чтобы восстановить градиент считаем все съеденные оси
        if axes is None:
            new_shape = [1 for i in out.data.shape]
        else:
            new_shape = list(out.data.shape)
            if axes is not list and axes is not tuple:
                axes = [axes]
            for i in axes:
                new_shape.insert(i, 1) 
        
        def _backward():
            self.grad += torch.tensor(out.grad).reshape(new_shape).broadcast_to(self.data.shape)
        
        out._backward = _backward
        
        return out
    
    def broadcast(self, new_shape):
        out = Value(self.data.broadcast_to(new_shape), (self,), 'broadcast')
        
        # Соберём, какие оси нужно будет просуммировать
        b_shape = [-1]*(len(new_shape) - len(self.data.shape)) + list(self.data.shape)
        axes = [i for i in range(len(new_shape)) if new_shape[i] != b_shape[i]]
        
        def _backward():
            self.grad += out.grad.sum(tuple(axes)).reshape(self.data.shape)
        out._backward = _backward
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
