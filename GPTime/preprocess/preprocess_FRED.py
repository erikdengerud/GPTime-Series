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
import sys

sys.path.append("")
from GPTime.config import cfg

logger = logging.getLogger(__name__)


def preprocess_FRED()->None:
    """
    Preprocess the FRED dataset.
    """

    # Read dummy file written by source.
    dummy_files = glob.glob(cfg.source.path.FRED + "/*")
    for f in dummy_files:
        X = np.load(f)

    # Write dummy preprocessed data
    Y = np.random.rand(10000, 200)
    filename="dummy_test.npy"
    path = os.path.join(cfg.preprocess.path.FRED, filename)
    np.save(path, Y)
    logger.debug("Preprocess FRED ran.")

if __name__ == "__main__":
    preprocess_FRED()