import numpy as np
import logging
import sys

sys.path.append("")

logger = logging.getLogger(__name__)

from GPTime.preprocess.preprocess_FRED import preprocess_FRED
from GPTime.preprocess.preprocess_M4 import preprocess_M4


def preprocess(cfg) -> None:
    """
    Preprocess raw data.
    """
    
    #preprocess_FRED(cfg_preprocess)
    preprocess_M4(
        raw_train_path=cfg.M4.raw_train,
        samples_per_json=cfg.samples_per_json,
        files_per_folder=cfg.files_per_folder,
        store_path=cfg.M4.store_path,
        )

    logger.debug("Preprocess ran.")
