import torch
import numpy as np

device = torch.device('cpu')

class Tensor:

    """ stores a single scalar Tensor and its gradient """

    def __init__(self, data, _children=(), _op=''):

        self.data = torch.tensor(data, dtype=torch.float32, device=device) if not isinstance(data, torch.Tensor) else data.to(device)
        self.grad = torch.zeros_like(self.data)

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):

        # (1) if other is not a Tensor, convert it to one
        other = other if isinstance(other, Tensor) else Tensor(other)

        # (2) create a new Tensor that is the sum of self and other
        out = Tensor(self.data + other.data, (self, other), '+')

        # (3) define the backward function for this operation
        def _backward():
            self_contrib = out.grad
            if self.grad.shape == () and self_contrib.shape != ():
                self.grad += self_contrib.sum()
            else:
                self.grad += self_contrib
            other_contrib = out.grad
            if other.grad.shape == () and other_contrib.shape != ():
                other.grad += other_contrib.sum()
            else:
                other.grad += other_contrib
        out._backward = _backward

        # (4) return the new Tensor
        return out

    def __mul__(self, other):

        # (1) if other is not a Tensor, convert it to one
        other = other if isinstance(other, Tensor) else Tensor(other)

        # (2) create a new Tensor that is the product of self and other
        out = Tensor(self.data * other.data, [self, other], '*')

        # (3) define the backward function for this operation
        def _backward():
            self_contrib = out.grad * other.data
            if self.grad.shape == () and self_contrib.shape != ():
                self.grad += self_contrib.sum()
            else:
                self.grad += self_contrib
            other_contrib = out.grad * self.data
            if other.grad.shape == () and other_contrib.shape != ():
                other.grad += other_contrib.sum()
            else:
                other.grad += other_contrib
        out._backward = _backward

        # (4) return the new Tensor
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data ** other.data, [self, other], '**')

        def _backward():
            self_contrib = out.grad * other.data * (self.data ** (other.data - 1))
            if self.grad.shape == () and self_contrib.shape != ():
                self.grad += self_contrib.sum()
            else:
                self.grad += self_contrib
            other_contrib = out.grad * torch.log(self.data) * (self.data ** other.data)
            if other.grad.shape == () and other_contrib.shape != ():
                other.grad += other_contrib.sum()
            else:
                other.grad += other_contrib
        out._backward = _backward

        return out
        # FIXED

    def relu(self):
        #FIXME
        out = Tensor(torch.relu(self.data), [self], 'ReLU')

        def _backward():
            self.grad += (self.data > 0).float() * out.grad
        out._backward = _backward

        # (4) return the new Tensor
        return out
        # FIXED

    def build_topo(self, visited=None, topo=None):
        if self not in visited:
            visited.add(self)
            for child in self._prev:
                child.build_topo(visited=visited, topo=topo)
            topo.append(self)
        return topo

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        topo = self.build_topo(topo=topo, visited=visited)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = torch.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


    def __neg__(self): # -self
        #FIXME
        return self * (-1)
        # FIXED

    def __radd__(self, other): # other + self
        #FIXME
        return self + other
        # FIXED

    def __sub__(self, other): # self - other
        #FIXME
        return self + (-other)
        # FIXED

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        #FIXME
        return self * other
        # FIXED

    def __truediv__(self, other): # self / other
        #FIXME
        return self * other ** (-1)
        # FIXED

    def __rtruediv__(self, other): # other / self
        #FIXME
        return other * self ** (-1)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    

# Additional mathematical functions with reverse-mode autodiff    

def log_d(dual_number: Tensor):
    # Implements log(x) with reverse-mode autodiff
    out = Tensor(torch.log(dual_number.data), (dual_number,), 'log')
    def _backward():
        contrib = (1 / dual_number.data) * out.grad
        if dual_number.grad.shape == () and contrib.shape != ():
            dual_number.grad += contrib.sum()
        else:
            dual_number.grad += contrib
    out._backward = _backward
    return out

def exp_d(dual_number: Tensor):
    # Implements exp(x) with reverse-mode autodiff
    out = Tensor(torch.exp(dual_number.data), (dual_number,), 'exp')
    def _backward():
        contrib = out.data * out.grad
        if dual_number.grad.shape == () and contrib.shape != ():
            dual_number.grad += contrib.sum()
        else:
            dual_number.grad += contrib
    out._backward = _backward
    return out

def sin_d(dual_number: Tensor):
    #FIXME
    out = Tensor(torch.sin(dual_number.data), (dual_number,), 'sin_d')
    def _backward():
      dual_number.grad += out.grad * (torch.cos(dual_number.data))
    out._backward = _backward
    return out
    # FIXED

def cos_d(dual_number: Tensor):
    #FIXME
    out = Tensor(torch.cos(dual_number.data), (dual_number,), 'cos_d')
    def _backward():
        dual_number.grad += out.grad * (-torch.sin(dual_number.data))
    out._backward = _backward
    return out
    # FIXED

def sigmoid_d(dual_number: Tensor):
    #FIXME
    out = Tensor(1/(1 + torch.exp(-dual_number.data)), (dual_number,), 'sigmoid_d')
    def _backward():
        contrib = out.grad * out.data * (1 - out.data)
        if dual_number.grad.shape == () and contrib.shape != ():
            dual_number.grad += contrib.sum()
        else:
            dual_number.grad += contrib
    out._backward = _backward
    return out
    # FIXED

def tanh_d(dual_number: Tensor):
    #FIXME
    out = Tensor(torch.tanh(dual_number.data), (dual_number,), 'tanh_d')
    def _backward():
        contrib = out.grad * (1 - out.data**2)
        if dual_number.grad.shape == () and contrib.shape != ():
            dual_number.grad += contrib.sum()
        else:
            dual_number.grad += contrib
    out._backward = _backward
    return out
    # FIXED

def tan_d(dual_number: Tensor):
    #FIXME
    out = Tensor(torch.tan(dual_number.data), (dual_number,), 'tan_d')
    def _backward():
        dual_number.grad += out.grad * (1 / torch.cos(dual_number.data)**2)
    out._backward = _backward
    return out
    # FIXED


def sqrt_d(dual_number: Tensor):
    #FIXME
    out = Tensor(torch.sqrt(dual_number.data), (dual_number,), 'sqrt_d')
    def _backward():
        dual_number.grad += out.grad * (0.5 / torch.sqrt(dual_number.data))
    out._backward = _backward
    return out
    # FIXED

def pow_d(dual_number: Tensor, power: int):
    #FIXME
    out = Tensor(dual_number.data ** power, (dual_number,), f'**{power}')
    def _backward():
        dual_number.grad += out.grad * (power * dual_number.data ** (power - 1))
    out._backward = _backward
    return out
    # FIXED

def softmax_d(dual_number: Tensor):
    #FIXME
    exp_vals = torch.exp(dual_number.data)
    out = Tensor(exp_vals / torch.sum(exp_vals), (dual_number,), 'softmax_d')
    def _backward():
        dual_number.grad += out.grad * (1)
    out._backward = _backward
    return out
    # FIXED

def matmul(a: Tensor, b: Tensor):
    out = Tensor(torch.matmul(a.data, b.data), (a, b), 'matmul')
    def _backward():
        if a.data.ndim == 2 and b.data.ndim == 1:
            a_contrib = torch.matmul(out.grad.unsqueeze(-1), b.data.unsqueeze(0))
            b_contrib = torch.matmul(a.data.mT, out.grad.unsqueeze(-1)).squeeze(-1)
        else:
            a_contrib = torch.matmul(out.grad, b.data.mT)
            b_contrib = torch.matmul(a.data.mT, out.grad)
        if a.grad.shape == () and a_contrib.shape != ():
            a.grad += a_contrib.sum()
        else:
            a.grad += a_contrib
        if b.grad.shape == () and b_contrib.shape != ():
            b.grad += b_contrib.sum()
        else:
            b.grad += b_contrib
    out._backward = _backward
    return out

def sum_d(dual_number: Tensor, axis=None):
    out = Tensor(torch.sum(dual_number.data, axis=axis), (dual_number,), 'sum')
    def _backward():
        dual_number.grad += torch.ones_like(dual_number.data) * out.grad
    out._backward = _backward
    return out

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weights for input to gates
        self.Wf = Tensor(torch.randn(hidden_size, input_size, device=device) * 0.01)
        self.Wi = Tensor(torch.randn(hidden_size, input_size, device=device) * 0.01)
        self.Wo = Tensor(torch.randn(hidden_size, input_size, device=device) * 0.01)
        self.Wc = Tensor(torch.randn(hidden_size, input_size, device=device) * 0.01)
        
        # Weights for hidden to gates
        self.Uf = Tensor(torch.randn(hidden_size, hidden_size, device=device) * 0.01)
        self.Ui = Tensor(torch.randn(hidden_size, hidden_size, device=device) * 0.01)
        self.Uo = Tensor(torch.randn(hidden_size, hidden_size, device=device) * 0.01)
        self.Uc = Tensor(torch.randn(hidden_size, hidden_size, device=device) * 0.01)
        
        # Biases
        self.bf = Tensor(torch.zeros(hidden_size, device=device))
        self.bi = Tensor(torch.zeros(hidden_size, device=device))
        self.bo = Tensor(torch.zeros(hidden_size, device=device))
        self.bc = Tensor(torch.zeros(hidden_size, device=device))
    
    def parameters(self):
        return [self.Wf, self.Wi, self.Wo, self.Wc, self.Uf, self.Ui, self.Uo, self.Uc, self.bf, self.bi, self.bo, self.bc]
    
    def forward(self, x, h_prev, c_prev):
        # x: (input_size,), h_prev: (hidden_size,), c_prev: (hidden_size,)
        
        # Forget gate
        f = sigmoid_d(matmul(self.Wf, x) + matmul(self.Uf, h_prev) + self.bf)
        
        # Input gate
        i = sigmoid_d(matmul(self.Wi, x) + matmul(self.Ui, h_prev) + self.bi)
        
        # Candidate values
        c_tilde = tanh_d(matmul(self.Wc, x) + matmul(self.Uc, h_prev) + self.bc)
        
        # Cell state
        c = f * c_prev + i * c_tilde
        
        # Output gate
        o = sigmoid_d(matmul(self.Wo, x) + matmul(self.Uo, h_prev) + self.bo)
        
        # Hidden state
        h = o * tanh_d(c)
        
        return h, c

class LSTM:
    def __init__(self, input_size, hidden_size, num_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
    
    def parameters(self):
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params
    
    def forward(self, x_seq):
        # x_seq: list of (input_size,) tensors, sequence length
        h = Tensor(np.zeros(self.hidden_size))
        c = Tensor(np.zeros(self.hidden_size))
        
        for x in x_seq:
            h, c = self.cells[0].forward(x, h, c)
        
        return h  # final hidden state

class Linear:
    def __init__(self, in_features, out_features):
        self.W = Tensor(torch.randn(out_features, in_features, device=device) * 0.01)
        self.b = Tensor(torch.zeros(out_features, device=device))
    
    def parameters(self):
        return [self.W, self.b]
    
    def forward(self, x):
        return matmul(self.W, x) + self.b
