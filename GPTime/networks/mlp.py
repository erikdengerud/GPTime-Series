import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import logging

logger = logging.getLogger(__name__)



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
        dropout:float=0.2,
        encode_frequencies:bool=False,
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
        self.dropout = dropout
        self.encode_frequencies = encode_frequencies

        self.one_hot = {
            "Y": np.array([1,0,0,0,0,0]),
            "Q": np.array([0,1,0,0,0,0]),
            "M": np.array([0,0,1,0,0,0]),
            "W": np.array([0,0,0,1,0,0]),
            "D": np.array([0,0,0,0,1,0]),
            "H": np.array([0,0,0,0,0,1]),
            "O": np.array([0,0,0,0,0,0]),
            "yearly": np.array([1,0,0,0,0,0]),
            "quarterly": np.array([0,1,0,0,0,0]),
            "monthly": np.array([0,0,1,0,0,0]),
            "weekly": np.array([0,0,0,1,0,0]),
            "daily": np.array([0,0,0,0,1,0]),
            "hourly": np.array([0,0,0,0,0,1]),
            }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        if self.encode_frequencies:
            self.layers.append(torch.nn.Linear(in_features=in_features+6, out_features=n_hidden))
        else:
            self.layers.append(torch.nn.Linear(in_features=in_features, out_features=n_hidden))
        self.dropout_layers.append(nn.Dropout(p=dropout))
        for i in range(1, self.n_layers-1):
            self.layers.append(torch.nn.Linear(in_features=n_hidden, out_features=n_hidden))
            self.dropout_layers.append(nn.Dropout(p=dropout))
        self.out_layer = torch.nn.Linear(in_features=n_hidden, out_features=out_features)
 
        self.init_weights()
        #logger.debug(f"Building model with frequency {self.frequency}")
        logger.debug(f"Input size of model: {in_features}.")
        logger.debug(f"Number of layers: {num_layers}.")
        logger.debug(f"Number of hidden units: {n_hidden}.")
        logger.debug(f"Using frequency encoding: {encode_frequencies}")

    def forward(self, x, mask, last_period, freq_str_arr):
        naive = torch.gather(x, 1, last_period.unsqueeze(1))
        #naive = 0
        if self.skip_connections:
            skip = 0
        if self.encode_frequencies:
            try:
                one_hot_freq = []
                for f in freq_str_arr:
                    #logger.debug(f"f[0]: {f[0]}")
                    one_hot_freq.append(self.one_hot[f[0]])
                #logger.debug(f"len(one_hot_freq): {len(one_hot_freq)}")
                #logger.debug(f"one_hot_freq[:10]: {one_hot_freq[:3]}")
                ohf_arr = torch.from_numpy(np.array(one_hot_freq)).to(self.device).double()
                x = torch.cat((x, ohf_arr), 1)
            except Exception as e:
                logger.debug(e)
                logger.debug(f"len(one_hot_freq): {len(one_hot_freq)}")
                logger.debug(f"one_hot_freq[:10]: {one_hot_freq[:3]}")
                logger.debug(f"x.shape: {x.shape}")
                ohf_arr = np.array(one_hot_freq)
                logger.debug(f"ohf_arr.shape: {ohf_arr.shape}")
                ohf_tens = torch.from_numpy(ohf_arr)
                logger.debug(f"ohf_tens.shape: {ohf_tens.shape}")
                #for ohf in one_hot_freq:
                #    logger.debug(ohf)
                #logger.debug(f"ohf_arr.shape: {ohf_arr.shape}")
        x = self.layers[0](x)
        x = self.dropout_layers[0](x)
        res = x
        for i, layer in enumerate(self.layers[1:], start=1):
            if (i+1) % self.res_block_size:
                x = res + F.relu(layer(x)) # This is supposed to be better since the signal can pass directly through ref https://arxiv.org/abs/1603.05027
                x = self.dropout_layers[i](x)
                res = x
                if self.skip_connections:
                    skip += x
            else:
                x = F.relu(layer(x))
                x = self.dropout_layers[i](x)
        if self.skip_connections:
            skip = skip + x
            out = self.out_layer(skip)
        else:
            out = self.out_layer(x)
        forecast = out + naive

        return forecast

    def init_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.out_layer.weight)
        """
        naive = torch.gather(x, 1, last_period.unsqueeze(1))
        for i, layer in enumerate(self.layers):
            #x = F.relu(layer(x))
            x = torch.relu(layer(x))
        out = self.out_layer(x)
        forecast = out + naive
        return forecast
        """
        #"""

        """
        naive = x[:, -1:] # A naive forecast as starting point 
        res = x     
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x))
        block_forecast = self.out_layer(x)
        forecast = naive  + block_forecast
        return forecast

        """


        """
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
        """

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
        skip_connections:bool=False,
        seasonal_naive:bool=False,
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


        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_features=in_features, out_features=n_hidden, bias=bias))
        for i in range(1, self.N-1):
            self.layers.append(torch.nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=bias))
        self.out = torch.nn.Linear(in_features=n_hidden, out_features=out_features, bias=bias)


        if self.residual == "ReZero":
            self.resweight = nn.Parameter(torch.zeros(int(num_layers/res_block_size)), requires_grad=True)
        
        #self.init_weights()

    def forward(self, x, mask, frequency):
        naive = x[:,-12].unsqueeze(1)
        print(naive.shape)
        for i, layer in enumerate(self.layers):
            #x = F.relu(layer(x))
            x = torch.relu(layer(x))
        out = self.out(x)
        forecast = out + naive
        return forecast
        """
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
            print(x.shape)
            out = self.out(x)
        forecast = naive + out
        return forecast
        """
    
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
