import torch.nn as nn

class AR(nn.Module):
    """
    AR(p) model.
    """
    def __init__(self, in_features:int, out_features:int=1, bias:bool=False):
        super(AR, self).__init__()
        self.memory = in_features
        self.bias = bias
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        self.init_weights()

    def forward(self, x):
        out = self.linear(x)
        return out

    def init_weights(self):
        nn.init.normal_(self.linear.weight, std=1e-3)
        if self.bias:
            nn.init.normal_(self.linear.bias, std=1e-6)