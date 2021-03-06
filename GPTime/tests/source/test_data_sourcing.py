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

from GPTime.source.data_sourcing import source
from GPTime.source.source_FRED import source_FRED
from GPTime.config import cfg

logger = logging.getLogger(__name__)


class TestSource(unittest.TestCase):
    """
    Test data sourcing.
    Test that all modules can be run. Specific tests for each submodule should be in 
    separate test files.
    """
    def setUp(self):
        pass

    def test_source_FRED(self):
        pass

    def test_files_written(self):
        pass

    def test_format(self):
        pass

if __name__ == "__main__":
    unittest.main(verbosity=2)