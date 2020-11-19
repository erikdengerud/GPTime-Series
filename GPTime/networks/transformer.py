import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeTrasformer(nn.Module):
    """
    N layer transformer with width W.
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, in_features:int, out_features:int=1, num_layers:int=5, width:int=32, residual:str="None", dropout=0.5):
        super(TimeTransformer, self).__init__()
        assert in_features == n_hidden, f"in_features, {in_features}, need to be similar to width, {width} "
        self.in_features = in_features
        self.out_features = out_features
        self.N = num_layers
        self.W = width
        self.memory = width
        self.residual = residual


        self.pos_encoder = PositionalEncoding(width, dropout)







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


class PositionalEncoding(nn.Module):
    """Create positional encodings using sine and cosine functions."""
    def __init__(self, lookback:int, d_model:int, dropout:float=0.1)->None:
        """Initializing the positional embedding.

        Args:
            lookback (int): Lookback of the model.
            d_model (int): Dimension of the model, number of dimensions in the TS.
            dropout (float, optional): Dropout. Defaults to 0.1.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(lookback, d_model)
        position = torch.arange(0, lookback, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(o, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """Forward function of the Positional Encoding layer.

        Args:
            x (torch.Tensor): Model input.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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