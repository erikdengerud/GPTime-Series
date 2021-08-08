import glob
import logging
import pandas as pd
import numpy as np
from typing import List, Dict
import os
import json
import sys

sys.path.append("")

logger = logging.getLogger(__name__)

def process_ts(row, index):
    """ Process a row in the M4 dfs. """
    data = row.to_numpy()
    data = data[~np.isnan(data)]

    observations = [{"value" : float(v)} for v in data]

    ts = {
        "frequency" : str(index)[0],
        "observations" : observations
    }
    return ts

def write_file(list_json, num_files_written, files_per_folder, write_path):
    """ Write list of jsons to file. """
    filename = f"processed_{num_files_written:>06}.json"
    curr_dir = f"dir{num_files_written // files_per_folder :04d}/"
    os.makedirs(os.path.join(write_path, curr_dir), exist_ok=True)
    with open(os.path.join(*[write_path, curr_dir, filename]), "w") as fp:
        json.dump(list_json, fp, sort_keys=True, indent=4, separators=(",", ": "))
        fp.close()

def preprocess_M4(raw_train_path, samples_per_json, files_per_folder, store_path) -> None:
    """
    Preprocess M4 into a training dataset format.
    """
    m4_fnames = glob.glob(os.path.join(raw_train_path, "*"))

    list_json:List[Dict] = []
    tot_processed = 0
    num_files_written = 0

    for fname in m4_fnames:
        logger.info(f"Processing file: {fname}")
        df = pd.read_csv(fname, index_col=0)
        for index, row in df.iterrows():
            try:
                ts = process_ts(row, index)
                list_json.append(ts)
                tot_processed += 1
            except Exception as e:
                logger.info(f"Failed to process ts at index: {index}")
                logger.info(e)


            if len(list_json) > samples_per_json:
                write_file(list_json, num_files_written, files_per_folder, store_path)
                num_files_written += 1
                list_json = []

            if tot_processed % 10000 == 0:
                logger.info(f"Processed {tot_processed} time series.")

    
    write_file(list_json, num_files_written, files_per_folder, store_path)
    logger.info("Done preprocessing M4.")
