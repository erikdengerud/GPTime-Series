import unittest
import sys
sys.path.append("")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtest import assert_vars_change
from torch.autograd import Variable

import torchtest


from GPTime.networks.ar import AR

class TestAR(unittest.TestCase):
    """
    Testing the AR model.
    """
    def test_all_variables_train(self):
        in_features=10
        out_features=1
        inputs = Variable(torch.randn(20, in_features))
        targets = Variable(torch.randn(20, out_features))
        batch = [inputs, targets]
        model = AR(in_features=in_features, out_features=out_features)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert_vars_change(
            model=model,
            loss_fn=F.mse_loss,
            optim=torch.optim.Adam(model.parameters()),
            batch=batch,
            device=device)
    
    def test_output_dimensons(self):
        in_features=10
        out_features=1
        inputs = Variable(torch.randn(20, in_features))
        targets = Variable(torch.randn(20, out_features))
        model = AR(in_features=in_features, out_features=out_features)
        outputs = model(inputs)
        self.assertEqual(targets.detach().cpu().numpy().shape, outputs.detach().cpu().numpy().shape)

    def test_test_suite(self):
        torch.manual_seed(1729)
        torchtest.setup()

        in_features=10
        out_features=1
        inputs = Variable(torch.randn(20, in_features))
        targets = Variable(torch.randn(20, out_features))
        batch = [inputs, targets]
        model = AR(in_features=in_features, out_features=out_features)

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
            test_vars_change=True,
            test_nan_vals=True,
            test_inf_vals=True,
            test_gpu_available=False,
            device=device
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)