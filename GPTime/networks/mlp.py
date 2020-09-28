import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    N layer multi layer perceptron with H hidden units in each layer.
    """
    def __init__(self, in_features:int, out_features:int=1, num_layers:int=5, n_hidden:int=32, bias:bool=True):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N = num_layers
        self.H = n_hidden
        self.memory = in_features
        self.bias=bias

        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(in_features=in_features if i==0 else n_hidden, out_features=n_hidden, bias=bias))
        self.out = nn.Linear(in_features=n_hidden, out_features=out_features, bias=bias)

        self.init_weights()

    def forward(self, x):
        for i in range(self.N-1):
            x = F.relu(self.layers[i](x))
        out = self.out(x)
        return out
    
    def init_weights(self):
        for i in range(self.N - 1):
            nn.init.normal_(self.layers[i].weight, std=1e-3)
            if self.bias:
                nn.init.normal_(self.layers[i].bias, std=1e-6)
        nn.init.normal_(self.out.weight, std=1e-3)
        if self.bias:
            nn.init.normal_(self.out.bias, std=1e-6)