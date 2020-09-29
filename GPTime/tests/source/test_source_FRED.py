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

from GPTime.source.source_FRED import source_FRED
from GPTime.config import cfg

logger = logging.getLogger(__name__)


class TestSourceFred(unittest.TestCase):
    """
    Test data sourcing from FRED.
    """
    def setUp(self):
        with open("GPTime/credentials.yml", "r") as ymlfile:
            self.credentials = Box(yaml.safe_load(ymlfile))
        # create and write file with ids to retrieve
        test_file = {
            "ATLSBUBEI" : "M",
            "ATLSBUBUI" : "M",
            "ATLSBUEGEI" : "M",
            "ATLSBUEGEP" : "M",
            "ATLSBUEGUI" : "M",
            "ATLSBUEGUP" : "M",
            "ATLSBUIREI" : "M",
            "ATLSBUIRUI" : "M",
            "ATLSBUSGEI" : "M",
            "ATLSBUSGUI" : "M",
            "ATLSBUSRGEP" : "M",
            "ATLSBUSRGUP" : "M",
        }
        with open("GPTime/tests/data/test_fred_list.json", "w") as fp:
            json.dump(test_file, fp)
        self.test_file = test_file
        for f in self.test_file.keys():
            if os.path.isfile(cfg.source.path.FRED.raw + f + ".json"):
                os.remove(cfg.source.path.FRED.raw + f + ".json")
        source_FRED(self.credentials.FRED, small_sample=True, id_freq_list_path="GPTime/tests/data/test_fred_list.json")

    def test_api_key(self):
        key = self.credentials.FRED.API_KEY_FED.key
        self.assertTrue(len(key) > 0)
        fred.key(key)
        category = fred.category()
        self.assertTrue(len(category) > 0)

    def test_files_written(self):
        for f in self.test_file.keys():
            self.assertTrue(os.path.isfile(cfg.source.path.FRED.raw + f+".json"))

    def test_format(self):
        for f in self.test_file.keys():
            with open(cfg.source.path.FRED.raw + f + ".json", "r") as fp:
                obs = json.load(fp)
            self.assertIsInstance(obs["id"], str)
            self.assertIsInstance(obs["source"], str)
            self.assertIsInstance(obs["frequency"], str)
            self.assertIsInstance(obs["values"], list)
            for val in obs["values"]:
                self.assertIsInstance(val, float)

if __name__ == "__main__":
    unittest.main(verbosity=2)