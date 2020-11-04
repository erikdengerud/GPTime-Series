"""
Forecasting metrics.
Functions scoring predictions.
"""
import numpy as np
import sys
import logging

sys.path.append("")
from GPTime.config import cfg

logger = logging.getLogger(__name__)


def MASE(actual: np.array, predicted: np.array, scale: np.array) -> float:
    """ Calculating Mean Absolute Scaled Error """
    """
    logger.info(f"np.abs(actual - predicted).shape: {np.abs(actual - predicted).shape}")
    logger.info(
        f"np.nanmean(np.abs(actual - predicted), axis=1).shape: {np.nanmean(np.abs(actual - predicted), axis=1).shape}"
    )
    a = np.nanmean(np.abs(actual - predicted), axis=1) / scale
    logger.info(
        f"np.nanmean(np.abs(actual - predicted), axis=1) / scale . shape: {a.shape}"
    )
    """
    return np.mean(np.nanmean(np.abs(actual - predicted), axis=1) / scale)


def SMAPE(actual: np.array, predicted: np.array) -> float:
    """ Calculating Symmetric Mean Absolute Prediction Error """
    return np.mean(
        np.nanmean(
            200.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)),
            axis=1,
        )
    )


def OWA(mase: float, smape: float, freq: str = "global") -> float:
    """ Calculating the Overall Weighted Average used in M4 """
    return (
        mase / cfg.scoring.owa.naive2.mase[freq]
        + smape / cfg.scoring.owa.naive2.smape[freq]
    ) / 2


# Below TODO
def mae(actual, predicted):
    return 0


def rmse(actual, predicted):
    return 0


def mape(actual, predicted):
    return 0


def wape(actual, predicted):
    return 0