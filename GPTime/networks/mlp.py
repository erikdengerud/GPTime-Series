import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

class MLP_orig(nn.Module):
    """
    N layer multi layer perceptron with H hidden units in each layer.
    """
    def __init__(
        self,
        in_features:int,
        out_features:int=1,
        num_layers:int=5,
        n_hidden:int=32,
        bias:bool=True,
        residual:str="None",
        res_block_size:int=1,
        forecast_horizon:int=1,
        skip_connections:bool=True,
        seasonal_naive:bool=True,
        ) -> None:

        super(MLP_orig, self).__init__()
        #assert in_features == n_hidden, f"in_features, {in_features}, need to be similar to n_hidden, {n_hidden} "
        self.in_features = in_features
        self.out_features = out_features
        self.N = num_layers
        self.H = n_hidden
        self.memory = in_features
        self.bias = bias
        self.residual = residual
        self.res_block_size = res_block_size
        self.forecast_horizon = forecast_horizon
        self.skip_connections = skip_connections
        self.seasonal_naive = seasonal_naive

        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(in_features=in_features if i==0 else n_hidden, out_features=n_hidden, bias=bias))
        self.out = nn.Linear(in_features=n_hidden, out_features=out_features, bias=bias)

        if self.residual == "ReZero":
            self.resweight = nn.Parameter(torch.zeros(int(num_layers/res_block_size)), requires_grad=True)
        
        self.init_weights()

    def forward(self, x, mask, frequency):
        if self.seasonal_naive:
            if self.forecast_horizon == 1:
                naive = x[[i for i in range(x.shape[0])], [-f for f in frequency]].unsqueeze(1)
            else:
                # We do not have to treat different frequencies here as the frequency should be
                # the same for all samples when using a multi-step forecast.
                period = x[:, -frequency:]
                num_periods = self.forecast_horizon // frequency
                naive = torch.cat([period for _ in range(num_periods+1)], dim=1)
                naive = naive[:, :self.forecast_horizon]
        else:
            naive = x[:, -1:]
        if self.skip_connections:
            skips = naive

        residual = x
        for i in range(self.N-1):
            if (i+1)%self.res_block_size == 0:# and i!=0:
                if self.residual == "ReZero":
                    x = x + self.resweight[int(i/self.res_block_size)] * (F.relu(self.layers[i](x)) )#* mask)
                elif self.residual == "Vanilla":
                    x = x +  F.relu(self.layers[i](x)) #* mask
                elif self.residual == None:
                    x = F.relu(self.layers[i](x)) #* mask
                if self.skip_connections:
                    skip = skip + x
            else:
                x = F.relu(self.layers[i](x)) #* mask
        if self.skip_connections:
            skip = skip + x 
            out = self.out(skip)
        else:
            out = self.out(x)
        forecast = naive + out
        return forecast
    
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

class MLP(torch.nn.Module): 
    """
    MLP 
    """
    def __init__(
        self,
        in_features,
        out_features,
        num_layers,
        n_hidden,
        residual:str=None,
        res_block_size:int=1,
        forecast_horizon:int=1,
        skip_connections:bool=False,
        seasonal_naive:bool=False,
        bias:bool=True,
        )->None:
        super(MLP, self).__init__()
        self.input_size = in_features
        self.output_size = out_features
        self.n_layers = num_layers
        self.layer_size = n_hidden
        self.res_block_size = res_block_size
        self.residual = residual
        self.frequency = 12
        self.skip_connections = skip_connections
        self.seasonal_naive = seasonal_naive
        self.memory = in_features
        
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_features=in_features, out_features=n_hidden))
        for i in range(1, self.n_layers-1):
            self.layers.append(torch.nn.Linear(in_features=n_hidden, out_features=n_hidden))
        self.out_layer = torch.nn.Linear(in_features=n_hidden, out_features=out_features)
 
        logger.debug(f"Building model with frequency {self.frequency}")

    def forward(self, x, mask, _):
        if self.seasonal_naive:
            naive = x[:, -self.frequency].unsqueeze(1) # Use the last period as the initial forec    ast for the next.
        else:
            naive = x[:, -1:] # A naive forecast as starting point 
        res = x     
        if self.skip_connections:
            skip = 0
        for i, layer in enumerate(self.layers):
            if (i+1)%self.res_block_size == 0:
                if self.residual == "Vanilla":
                    x = torch.relu(layer(x)) + res
                    res = x
                    if self.skip_connections:
                        skip = skip + x
                else:
                    x = torch.relu(layer(x))
            else:
                x = torch.relu(layer(x))
        if self.skip_connections:
            skip = skip + x
            block_forecast = self.out_layer(skip)
        else: 
            block_forecast = self.out_layer(x)
        forecast = naive  + block_forecast
        return forecast

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
    x = torch.randn(3, 10)
    mask = np.zeros((3, 10))
    mask[:, 5:] = 1.0
    mask = torch.from_numpy(mask).float()
    print(x)
    print(mask)
    #print(mask.shape)
    #print(F.relu(nn.Linear(10,10)(x)).shape)
    out = mlp(x, mask, 1)
    print(out)