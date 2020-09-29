import unittest
import logging
import fred
import glob
import os
import yaml
import json
from box import Box
import sys
sys.path.append("")

from GPTime.preprocess.preprocess import preprocess
from GPTime.preprocess.preprocess_FRED import preprocess_FRED
from GPTime.config import cfg

logger = logging.getLogger(__name__)


class TestPreprocessing(unittest.TestCase):
    """
    Test data preprocessing.
    Test that all modules can be run. Specific tests for each submodule should be in 
    separate test files.
    """
    def test_preprocess_FRED_module(self):
        preprocess_FRED()


if __name__ == "__main__":
    unittest.main(verbosity=2)