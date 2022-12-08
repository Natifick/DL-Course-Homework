import random
from framework.autograd import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


def autoSoftmax(x: Value, dim=0):
    """ Simple softmax implementation with my own autograd """
    e = x.exp()
    print(e)
    return e / e.sum(dim)


class Neuron(Module):
    #                  nin hao ma
    def __init__(self, nin, nonlin=True):
        self.w = [Value(1) for i in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        mid = sum([self.w[i] * x[i] for i in range(len(x))]) + self.b
        act = mid.relu() if self.nonlin else mid
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, kwargs['nonlin']) for i in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for n in self.neurons:
            params += n.parameters()
        return params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]
        
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            params += l.parameters()
        return params

    def __repr__(self):
        repr = '\n'.join(str(layer) for layer in self.layers)
        return f"MLP of [{repr}]"