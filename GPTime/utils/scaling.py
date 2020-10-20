

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
        if len(ts) <= period:
            period = 1
        scale = np.mean(np.abs((ts - np.roll(ts, shift=period))[period:]))
        if scale == 0:
            scale = ts[0] if ts[0] != 0 else 1
        assert scale > 0, f"Scale is not positive! {ts}, scale: {scale}"
        return scale

class MASEScaler(object):
    """
    Scaler as in sklearn.preprocessing. Does not handle 
    """
    def __init__(self, seasonality:bool=True):
        self.seasonality = seasonality
        self.periods = {
            "Y" : 1,
            "Q" : 4,
            "M" : 12,
            "W" : 1,
            "D" : 1,
            "H" : 24,
            "O" : 1
            }

    def fit(X, period):
        return self.partial_fit(X, freq)

    def partial_fit(X, freq):
        period = self.periods[freq]
        if len(X) <= period:
            period = 1
        scale = np.mean(np.abs((ts - np.roll(ts, shift=period))[period:]))
        if scale == 0:
            scale = ts[0] if ts[0] != 0 else 1
        assert scale > 0, f"Scale is not positive! {ts}, scale: {scale}"
        self.scale_ = scale
        return self
    
    def transform(X):
        X /= self.scale_
        return X

    def fit_transform(X, freq):
        pass

    def get_params(X):
        return X


    def inverse_transform(X):
        pass
    
    def set_params(**params):
        pass
    



if __name__ == "__main__":
    x = np.arange(20, step=2)
    x = np.array([3900, 4500, 4200, 4000, 4000, 3900, 4200, 4200, 4200, 5000, 4400])
    x = np.array([3540, 3560, 3560, 3560, 3520, 3290, 3360, 3390, 3340, 3330, 3340, 3310])
    logger.info(len(x))
    logger.info(x)
    logger.info(MASEscale(x, "Q"))
    logger.info(MASEscale(x, "Y"))
    logger.info(MASEscale(x, "M"))
    logger.info(MASEscale(x, "W"))
    logger.info(MASEscale(x, "D"))
    logger.info(MASEscale(x, "H"))