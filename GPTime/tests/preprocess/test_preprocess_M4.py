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
from GPTime.config import cfg

logger = logging.getLogger(__name__)


class TestPreprocessingM4(unittest.TestCase):
    """
    Test data preprocessing of M4.
    """
    def setUp(self):
        pass

    def test_api_key(self):
        pass

    def test_files_written(self):
        pass

    def test_format(self):
        pass
if __name__ == "__main__":
    unittest.main(verbosity=2)