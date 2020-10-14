

import logging
import numpy as np
from typing import Tuple
import sys

sys.path.append("")

from GPTime.config import cfg

logger = logging.getLogger(__name__)


def MASEscale(ts:np.array, freq:str) -> float:#Tuple[np.array, float]:
        """ Scale time series using scaling from MASE """
        d = {
            "Y" : "yearly",
            "Q" : "quarterly",
            "M" : "monthly",
            "W" : "weekly",
            "D" : "daily",
            "H" : "hourly",
            "O" : "other"
            }
        period = cfg.scoring.m4.periods[d[freq]]
        #logger.info(period)
        scale = np.mean(np.abs((ts - np.roll(ts, shift=period))[period:]))
        #ts_scaled = ts / scale
        return scale#ts_scaled, scale

if __name__ == "__main__":
    x = np.arange(20, step=2)
    logger.info(len(x))
    logger.info(x)
    logger.info(MASEscale(x, "Q"))