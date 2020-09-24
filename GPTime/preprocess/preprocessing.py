"""
Preprocess: Read from staging DB/files, use asserts to validate key staged data 
properties, engineer features, transform, and save to new DB table(s) or files.
"""

"""
Preprocess: Read the dummy table/file written by source. Don’t do anything with it. 
Write dummy preprocessed data for predict.
"""

import numpy as np
import logging
import sys

sys.path.append("")
from GPTime.config import cfg

logger = logging.getLogger(__name__)

from GPTime.preprocess.preprocess_FRED import preprocess_FRED

def preprocess()->None:
    """
    Preprocess raw data.
    """
    preprocess_FRED()

    logger.debug("Preprocess ran.")



if __name__ == "__main__":
    preprocess()