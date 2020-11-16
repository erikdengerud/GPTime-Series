import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    N layer multi layer perceptron with H hidden units in each layer.
    """
    def __init__(self, in_features:int, out_features:int=1, num_layers:int=5, n_hidden:int=32, bias:bool=True, residual:str="None", res_block_size:int=1):
        super(MLP, self).__init__()
        assert in_features == n_hidden, f"in_features, {in_features}, need to be similar to n_hidden, {n_hidden} "
        self.in_features = in_features
        self.out_features = out_features
        self.N = num_layers
        self.H = n_hidden
        self.memory = in_features
        self.bias = bias
        self.residual = residual
        self.res_block_size = res_block_size

        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(in_features=in_features if i==0 else n_hidden, out_features=n_hidden, bias=bias))
        self.out = nn.Linear(in_features=n_hidden, out_features=out_features, bias=bias)

        if self.residual == "ReZero":
            self.resweight = nn.Parameter(torch.zeros(int(num_layers/res_block_size)), requires_grad=True)
        
        self.init_weights()

    def forward(self, x, mask):
        residual = x
        for i in range(self.N-1):
            if (i+1)%self.res_block_size == 0:# and i!=0:
                if self.residual == "ReZero":
                    x = x + self.resweight[int(i/self.res_block_size)] * (F.relu(self.layers[i](x)) * mask)
                elif self.residual == "Vanilla":
                    x = x +  F.relu(self.layers[i](x)) * mask
                elif self.residual == None:
                    x = F.relu(self.layers[i](x)) * mask
            else:
                x = F.relu(self.layers[i](x)) * mask
        out = self.out(x)
        return out
    
    def init_weights(self):
        for i in range(self.N - 1):
            nn.init.normal_(self.layers[i].weight, std=1e-3)
            #nn.init.xavier_normal_(self.layers[i].weight)
            if self.bias:
                nn.init.normal_(self.layers[i].bias, std=1e-6)
        nn.init.normal_(self.out.weight, std=1e-3)
        #nn.init.xavier_normal_(self.layers[i].weight)
        if self.bias:
            nn.init.normal_(self.out.bias, std=1e-6)



if __name__ == "__main__":
    mlp = MLP(
        in_features=10,
        out_features=1,
        num_layers=10,
        n_hidden=10,
        bias=True,
        residual="ReZero",
        res_block_size=3,
        )
    import numpy as np
    x = torch.randn(10)
    mask = np.zeros(10)
    mask[5:] = 1.0
    mask = torch.from_numpy(mask).float()
    print(x)
    print(mask)
    #print(mask.shape)
    #print(F.relu(nn.Linear(10,10)(x)).shape)
    out = mlp(x, mask)
    print(out)