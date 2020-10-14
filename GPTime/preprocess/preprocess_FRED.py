"""
Preprocess: Read from staging DB/files, use asserts to validate key staged data 
properties, engineer features, transform, and save to new DB table(s) or files.
"""

"""
Preprocess: Read the dummy table/file written by source. Don’t do anything with it. 
Write dummy preprocessed data for predict.
"""

import numpy as np
from typing import Dict, List, Tuple
import glob
import logging
import os
import json
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import sys

sys.path.append("")
from GPTime.config import cfg

logger = logging.getLogger(__name__)

def find_frequency(freq_str:str) -> Tuple[str,str]:
    """
    Deriving correct frequeny of the frequency string in FRED.
    """
    if "yearly" in freq_str.lower():
        freq = "yearly"
        freq_short = "Y"
    elif "annual" in freq_str.lower():
        freq = "yearly"
        freq_short = "Y"
    elif "quarterly" in freq_str.lower():
        freq = "quarterly"
        freq_short = "Q"
    elif "monthly" in freq_str.lower():
        freq = "monthly"
        freq_short = "M"
    elif "weekly" in freq_str.lower():
        freq = "weekly"
        freq_short = "W"
    elif "daily" in freq_str.lower():
        freq = "daily"
        freq_short = "D"
    elif "hourly" in freq_str.lower():
        freq = "hourly"
        freq_short = "H"
    else:
        freq = "other"
        freq_short = "O"
    return freq, freq_short

def process_ts(ts:Dict) -> Tuple[List[Dict], bool, str]:
    """
    Processing time series. Splitting by missing values and keeping sub series if the sub series are long enough.
    """
    sub_lists = []
    curr = []
    contains_na = False
    for obs in ts["observations"]:
        if obs["value"] != '.':
            curr.append(obs)
        else:
            sub_lists.append(curr)
            curr = []
            contains_na = True
    sub_lists.append(curr)

    freq, freq_short = find_frequency(ts["frequency"])

    sub_lists = [l for l in sub_lists if len(l) > cfg.preprocess.ts.min_length[freq]]

    new_jsons = []
    
    for l in sub_lists:
        new_ts = {
            "frequency" : freq_short,
            "observations" : l,
        }
        new_jsons.append(new_ts)

    return new_jsons, contains_na, freq

def preprocess_FRED(account_url:str, container_name:str, dates=False)->None:
    """
    Preprocess the FRED dataset.
    """
    # Azure clients
    container_client = ContainerClient(account_url=account_url, container_name=container_name)
    all_blobs = container_client.list_blobs()
    raw_blob_names = [b["name"] for b in all_blobs if "raw" in b["name"]]

    list_json = []
    num_files_written = 0
    num_ts = 0
    num_contains_na = 0
    num_ts_processed = 0
    all_frequencies:Dict= {}
    
    for i, bname in enumerate(raw_blob_names):
        print(f"Name of blob: {bname}")
        blob_client = BlobClient(account_url=account_url, container_name=container_name, blob_name=bname)
        blob_json = json.loads(blob_client.download_blob().readall())
        
        for ts in blob_json:
            try:
                processed_jsons, contains_na, freq = process_ts(ts)
                num_ts += len(processed_jsons)
                num_contains_na += contains_na
                num_ts_processed += 1
                if freq in all_frequencies.keys():
                    all_frequencies[freq] += 1
                else:
                    all_frequencies[freq] = 1

                list_json.extend(processed_jsons)

                if len(list_json) > cfg.source.samples_per_json:
                    filename = f"processed_{num_files_written:>06}.json"
                    if num_files_written % cfg.source.files_per_folder == 0:
                        curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
                        os.makedirs(os.path.join(cfg.preprocess.path.FRED, curr_dir), exist_ok=True)
                    with open(os.path.join(*[cfg.preprocess.path.FRED, curr_dir, filename]), "w") as fp:
                        json.dump(list_json, fp, sort_keys=True, indent=4, separators=(",", ": "))
                        fp.close()

                    num_files_written += 1
                    list_json = []
            except Exception as e:
                logger.warning(e)

        if i % 100 == 0:
            logger.info(f"Processed {i/len(raw_blob_names)*100:.2f}\% of blobs.")
            logger.info(f"Currently have {num_ts} time series")
            logger.info(f"Of the {num_ts_processed} time series processed, {num_contains_na/num_ts_processed*100:.2f}% contains missing values.")

    filename = f"processed_{num_files_written:>06}.json"
    if num_files_written % cfg.source.files_per_folder == 0:
        curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
        os.makedirs(os.path.join(cfg.preprocess.path.FRED, curr_dir), exist_ok=True)
    with open(os.path.join(*[cfg.preprocess.path.FRED, curr_dir, filename]), "w") as fp:
        json.dump(list_json, fp, sort_keys=True, indent=4, separators=(",", ": "))
        fp.close()

    logger.info(f"Processed {num_ts_processed} files")
    logger.info(f"Currently have {num_ts} time series")
    logger.info(f"Of the {num_ts_processed} time series processed, {num_contains_na/num_ts_processed*100:.2f}% contains missing values.")
    logger.info("Proportion of frequencies: ")
    tot_freq = sum(all_frequencies.values())
    for k, v in all_frequencies.items():
        logger.info(f"{k} : {v/tot_freq*100:.2f}")
    logger.info("Done preprocessing FRED.")



if __name__ == "__main__":
    preprocess_FRED(account_url="https://tsdatasets.blob.core.windows.net/", container_name="fred")