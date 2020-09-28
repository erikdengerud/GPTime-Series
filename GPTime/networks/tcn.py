import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from typing import List

# Causal convolution
class DilatedCausalConv(nn.Conv1d):
    """ https://github.com/pytorch/pytorch/issues/1333 """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation:int=1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(DilatedCausalConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(DilatedCausalConv, self).forward(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        stride:int,
        dilation:int,
        dropout:float=0.2,
        bias:bool=True,
        apply_weight_norm:bool=True
        )->None:
        super(TemporalBlock, self).__init__()

        self.dcc1 = DilatedCausalConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,)

        self.dropout1 = nn.Dropout(dropout)

        self.dcc2 = DilatedCausalConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,)

        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        if apply_weight_norm:
            self.dcc1 = weight_norm(self.dcc1)
            self.dcc2 = weight_norm(self.dcc2)

        self.init_weights()
    
    def forward(self, x):
        out = F.relu(self.dcc1(x))
        out = self.dropout1(out)
        out = F.relu(self.dcc2(out))
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        out = F.relu(out + res)
        return out

    def init_weights(self):
        nn.init.normal_(self.dcc1.weight, std=1e-3)
        nn.init.normal_(self.dcc2.weight, std=1e-3)
        if self.dcc1.bias is not None:
            nn.init.normal_(self.dcc1.bias, std=1e-6)
        if self.dcc1.bias is not None:
            nn.init.normal_(self.dcc2.bias, std=1e-6)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, std=1e-3)
            nn.init.normal_(self.downsample.bias, std=1e-6)


class TCN(nn.Module):
    def __init__(
        self,
        in_channels:int,
        channels:List[int],
        kernel_size:int=2,
        dropout:float=0.2
        )->None:
        super(TCN, self).__init__()
        self.memory = 1
        self.layers = nn.ModuleList()
        for i in range(len(channels)):
            dilation_size = 2 ** i
            self.layers.append(
                TemporalBlock(
                    in_channels=in_channels if i==0 else channels[i-1],
                    out_channels=channels[i],
                    stride=1,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                    )
                )

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x