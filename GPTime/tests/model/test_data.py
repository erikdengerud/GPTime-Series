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

class TestDataLoader(unittest.TestCase):
    """
    Testing the dataloader.
    """

    def test_shape_sample_linear(self):
        pass

    def test_shape_sample_convolution(self):
        pass

    def test_sampling_of_sampled_ids(self):
        pass

    def test_multi_task_sample(self):
        """ Test samples with frequency as part of sample. """
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)