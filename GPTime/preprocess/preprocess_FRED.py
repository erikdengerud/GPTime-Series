"""
Preprocess: Read from staging DB/files, use asserts to validate key staged data 
properties, engineer features, transform, and save to new DB table(s) or files.
"""

"""
Preprocess: Read the dummy table/file written by source. Don’t do anything with it. 
Write dummy preprocessed data for predict.
"""

import numpy as np
import glob
import logging
import os
import json
import sys

sys.path.append("")
from GPTime.config import cfg

logger = logging.getLogger(__name__)


def preprocess_FRED()->None:
    """
    Preprocess the FRED dataset.
    """

    # Read dummy file written by source.
    num_read = 0
    dummy_dirs = glob.glob(cfg.source.path.FRED.raw + "*")
    for d in dummy_dirs:
        dummy_files = glob.glob(d + "/*")
        for fname in dummy_files:
            with open(fname, "r") as f:
                in_file = json.load(f)
                num_read += 1
    logger.info(f"Read {num_read} files.")

    # Write dummy preprocessed data
    num_preprocessed = 0
    for i in range(10000):
        if num_preprocessed % 1000 == 0:
                curr_dir = f"dir{num_preprocessed // 1000 :03d}/"
                os.makedirs(cfg.preprocess.path.FRED + curr_dir, exist_ok=True)
        out = {
            "source" : "FRED",
            "id": f"{i:03d}",
            "frequency" : np.random.choice(["Y", "Q", "M", "W", "D", "H"]),
            "values" : list(np.random.rand(100)),
        }
        filename = f"{i:04d}.json"
        with open(cfg.preprocess.path.FRED+ curr_dir + filename, "w") as fp:
            json.dump(out, fp)
        num_preprocessed += 1
    logger.info(f"Wrote {num_preprocessed} preprocessed files.")
    logger.debug("Preprocess FRED ran.")

if __name__ == "__main__":
    preprocess_FRED()