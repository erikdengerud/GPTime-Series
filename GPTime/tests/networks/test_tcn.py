import unittest
import sys
sys.path.append("")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtest import assert_vars_change
from torch.autograd import Variable

import torchtest

import numpy as np
import logging

logger = logging.getLogger(__name__)


from GPTime.networks.tcn import TCN, DilatedCausalConv, TemporalBlock

class TestTCN(unittest.TestCase):
    """
    Testing the TCN model.
    """
    def test_dilated_causal_conv_1d_convolution(self):
        dcc = DilatedCausalConv(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False
        ).double()
        with torch.no_grad():
            dcc.weight[0][0] = 1
        inputs = torch.from_numpy(np.array([1 for i in range(10)])).view(1,1,-1).double()
        outputs = dcc(inputs).detach().cpu().view(-1).numpy()
        self.assertEqual(list(outputs), [1, 2, 3, 3, 3, 3, 3, 3, 3, 3])
    
    def test_dilated_causal_conv_1d_dilation(self):
        dcc = DilatedCausalConv(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            dilation=3,
            bias=False
        ).double()
        with torch.no_grad():
            dcc.weight[0][0] = 1
        inputs = torch.from_numpy(np.array([1 for i in range(10)])).view(1,1,-1).double()
        outputs = dcc(inputs).detach().cpu().view(-1).numpy()
        self.assertEqual(list(outputs), [1, 1, 1, 2, 2, 2, 3, 3, 3, 3])

    def test_temporal_block_convolution(self):
        tb = TemporalBlock(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            apply_weight_norm=False
        ).double()
        tb.eval()
        with torch.no_grad():
            tb.dcc1.weight[0][0] = 1
            tb.dcc2.weight[0][0] = 1
        inputs = torch.from_numpy(np.array([1 for i in range(10)])).view(1,1,-1).double()
        outputs = tb(inputs).detach().cpu().view(-1).numpy()
        self.assertEqual(list(outputs), [2, 4, 7, 9, 10, 10, 10, 10, 10, 10])

    def test_temporal_block_dilation(self):
        tb = TemporalBlock(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            dilation=3,
            bias=False,
            apply_weight_norm=False
        ).double()
        tb.eval()
        with torch.no_grad():
            tb.dcc1.weight[0][0] = 1
            tb.dcc2.weight[0][0] = 1
        inputs = torch.from_numpy(np.array([1 for i in range(10)])).view(1,1,-1).double()
        outputs = tb(inputs).detach().cpu().view(-1).numpy()

        self.assertEqual(list(outputs), [2, 2, 2, 4, 4, 4, 7, 7, 7, 9])


    def test_all_variables_train(self):
        in_features=1
        out_features=1
        inputs = Variable(torch.randn(100, in_features, 20))
        targets = Variable(torch.randn(100, out_features, 20))
        batch = [inputs, targets]
        model = TCN(
            in_channels=in_features,
            channels=[10,out_features],
            kernel_size=2,
            )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert_vars_change(
            model=model,
            loss_fn=F.mse_loss,
            optim=torch.optim.Adam(model.parameters()),
            batch=batch,
            device=device)
    
    def test_output_dimensons(self):
        in_features=1
        out_features=1
        inputs = Variable(torch.randn(100, in_features, 20))
        targets = Variable(torch.randn(100, out_features, 20))
        batch = [inputs, targets]
        model = TCN(
            in_channels=in_features,
            channels=[10,out_features],
            kernel_size=2,
            )
        outputs = model(inputs)
        self.assertEqual(targets.detach().cpu().numpy().shape, outputs.detach().cpu().numpy().shape)

    def test_test_suite(self):
        torch.manual_seed(1729)
        torchtest.setup()
        in_features=1
        out_features=1
        inputs = Variable(torch.randn(100, in_features, 20))
        targets = Variable(torch.randn(100, out_features, 20))
        batch = [inputs, targets]
        model = TCN(
            in_channels=in_features,
            channels=[10,out_features],
            kernel_size=2,
            )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torchtest.test_suite(
            model=model,
            loss_fn=F.mse_loss,
            optim=torch.optim.Adam(model.parameters()),
            batch=batch,
            output_range=None,
            train_vars=None,
            non_train_vars=None,
            test_output_range=False,
            test_vars_change=False,
            test_nan_vals=True,
            test_inf_vals=True,
            test_gpu_available=False,
            device=device
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)