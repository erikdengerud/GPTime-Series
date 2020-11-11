import torch
from torch.utils.data import Dataset

import logging
import glob
import json
import os
import numpy as np
import time
from typing import Dict, List, Tuple
import importlib
import sys

sys.path.append("")

from GPTime.config import cfg
from GPTime.utils.scaling import MASEScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

def prepare_ts(ts: Dict):
    vals = np.array([float(obs["value"]) for obs in ts["observations"]])
    if np.var(vals) < 0.0001:
        logger.info(np.var(vals))
        logger.info(ts)

    freq_str = ts["frequency"]
    freq_int = cfg.dataset.scaling.periods[freq_str]
    values_scaled = MASEScaler().fit_transform(vals, freq=freq_int)

    if len(values_scaled) < 13:
        values_scaled = np.array([])
    else:
        values_scaled = values_scaled[-13:]

    return values_scaled

def make_simple():
        
    # Read data into memory
    all_ts = []
    dirs = glob.glob("/work/erikde/data/processed/FRED_small" + "/*")
    freqs = {}
    for d in dirs:
        logger.info(f"Loading dir: {d}")
        fnames = glob.glob(os.path.join(d, "*"))
        for fname in fnames:
            with open(fname, "r") as fp:
                ts_list = json.load(fp)
                fp.close()
            for ts in ts_list:

                vals = np.array([float(obs["value"]) for obs in ts["observations"]])
                if len(set(vals)) < 2:
                    values_scaled=np.array([])
                else:
                    
                    freq_str = ts["frequency"]
                    freq_int = cfg.dataset.scaling.periods[freq_str]
                    #values_scaled = MASEScaler().fit_transform(vals, freq=freq_int)
                    values_scaled = MaxAbsScaler().fit_transform(vals.reshape(1,-1).T).flatten()
                    if len(values_scaled) < 13:
                        values_scaled = np.array([])
                    else:
                        values_scaled = values_scaled[-13:]


                if values_scaled.size:
                    if np.mean(values_scaled) < 3*10**4:
                        all_ts.append(values_scaled)
                        if ts["frequency"] in freqs:
                            freqs[ts["frequency"]] += 1
                        else:
                            freqs[ts["frequency"]] = 1
    all_ts = np.array(all_ts)
    logger.info(all_ts.shape)
    np.save("FRED_small.npy", all_ts)
    for k in freqs.keys():
        logger.info(f"{k} : {freqs[k]}")


if __name__ == "__main__":
    make_simple()