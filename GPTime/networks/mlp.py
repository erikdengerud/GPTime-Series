import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    N layer multi layer perceptron with H hidden units in each layer.
    """
    def __init__(self, in_features:int, out_features:int, num_layers:int, n_hidden:int, bias:bool=True):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N = num_layers
        self.H = n_hidden
        self.memory = in_features
        self.bias=bias

        self.layers = nn.ModuleList()
        for i in range(self.N-1):
            self.layers.append(nn.Linear(in_features=self.in_features if i==0 else self.H, out_features=self.H, bias=self.bias))
        self.out = nn.Linear(in_features=self.H, out_features=self.out_features, bias=self.bias)

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

if __name__ == "__main__":
    mlp = MLP(in_features=10, out_features=1, num_layers=3, n_hidden=8)
    print(mlp)
