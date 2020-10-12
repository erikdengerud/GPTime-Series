"""
Preprocess: Read from staging DB/files, use asserts to validate key staged data 
properties, engineer features, transform, and save to new DB table(s) or files.
"""

"""
Preprocess: Read the dummy table/file written by source. Don’t do anything with it. 
Write dummy preprocessed data for predict.
"""


import glob
import logging
import pandas as pd
import numpy as np
from typing import List, Dict
import os
import json
#from datetime import date
import datetime
import sys

sys.path.append("")
from GPTime.config import cfg

logger = logging.getLogger(__name__)

def check_frequency(start_year, periods, freq):
    """
    Validate frequency.
    """
    derived_freq = False
    if freq == "Y":
        end_year = start_year + periods
    elif freq == "Q":
        end_year = start_year + periods // 4
    elif freq == "M":
        end_year = start_year + periods // 12
    elif freq == "W":
        end_year = start_year + periods // 52
    elif freq == "D":
        end_year = start_year + periods // 365
    elif freq == "H":
        end_year = start_year + periods // (24*365)

    while end_year > 2020:
        derived_freq = True
        if freq == "Y":
            freq = "Q"
            end_year = start_year + periods // 4
        elif freq == "Q":
            freq = "M"
            end_year = start_year + periods // 12
        elif freq == "M":
            freq ="W"
            end_year = start_year + periods // 52
        elif freq == "W":
            freq ="D"
            end_year = start_year + periods // 365
        elif freq == "D":
            freq ="H"
            end_year = start_year + periods // (24*365)
        else:
            break

    return freq, derived_freq


def preprocess_M4(local:bool=True, azure_account_url:str=None, azure_container_name:str=None) -> None:
    """
    Preprocess M4 into a training dataset format.
    """
    if not local:
        assert azure_account_url is not None, "azure_account_url is None"
        assert azure_container_name is not None, "azure_container_name is None"
    # get raw data
    if local:
        m4_fnames = glob.glob(cfg.source.path.M4.raw_train + "*")
    #logger.info(m4_fnames)

    meta_df = pd.read_csv(cfg.source.path.M4.meta, index_col=0)
    list_json:List[Dict] = []
    tot_processed = 0
    num_files_written = 0
    tot_derived_freq = 0
    # preprocess

    for fname in m4_fnames:
        logger.info(fname)
        df = pd.read_csv(fname, index_col=0)
        for index, row in df.iterrows():
            try:
                meta_ts = meta_df.loc[index]

                data = row.to_numpy()
                data = data[~np.isnan(data)]

                # Check that the frequency is correct
                start_date = pd.Timestamp(meta_ts["StartingDate"])
                if start_date.year > 2018:
                    start_date = start_date.replace(year=start_date.year-100)
                freq = meta_ts["SP"][0]
                
                freq, derived_freq = check_frequency(start_date.year, periods=len(data), freq=freq)
                
                date_range = pd.date_range(start=start_date, periods=len(data), freq=freq).strftime("%Y-%m-%d %H:%M:%S")
                observations = [{"date" : d, "value" : float(v)} for d, v in zip(date_range, data)]

                ts = {
                    "frequency" : str(index)[0],
                    "derived_freq" : freq,
                    "derived_freq_flag" : derived_freq,
                    "observations" : observations
                }
                list_json.append(ts)
                tot_processed += 1
                tot_derived_freq += derived_freq
                
            except Exception as e:
                meta_ts = meta_df.loc[index]
                logger.info(f"Problem with series {index}")
                data = row.to_numpy()
                data = data[~np.isnan(data)]
                logger.info(f"Length of row is {len(data)}")
                logger.info(f"Starting date {meta_ts['StartingDate']}")
                logger.info(f"Frequency {meta_ts['SP']}")
                logger.warning(e)

            if len(list_json) > cfg.source.samples_per_json:
                filename = f"processed_{num_files_written:>06}.json"
                if num_files_written % cfg.source.files_per_folder == 0:
                    curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
                    os.makedirs(os.path.join(cfg.preprocess.path.M4, curr_dir), exist_ok=True)
                with open(os.path.join(*[cfg.preprocess.path.M4, curr_dir, filename]), "w") as fp:
                    json.dump(list_json, fp, sort_keys=True, indent=4, separators=(",", ": "))
                    fp.close()

                num_files_written += 1
                list_json = []

            if tot_processed % 10000 == 0:
                logger.info(f"Processed {tot_processed} time series.")

    filename = f"processed_{num_files_written:>06}.json"
    if num_files_written % cfg.source.files_per_folder == 0:
        curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
        os.makedirs(os.path.join(cfg.preprocess.path.M4, curr_dir), exist_ok=True)
    with open(os.path.join(*[cfg.preprocess.path.M4, curr_dir, filename]), "w") as fp:
        json.dump(list_json, fp, sort_keys=True, indent=4, separators=(",", ": "))
        fp.close()
    logger.info(f"Derived frequency of {tot_derived_freq}")


if __name__ == "__main__":
    preprocess_M4()