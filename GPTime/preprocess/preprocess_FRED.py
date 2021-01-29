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

def process_ts(ts:Dict, min_lengths:Dict, remove_zero:bool=True, remove_zero_treshold:float=0.8, num_distinct:int=5) -> Tuple[List[Dict], bool, str]:
    """
    Processing time series. Splitting by missing values and keeping sub series if the sub series are long enough.
    """
    sub_lists = []
    curr = []
    contains_na = False
    not_enough_distinct = 0
    too_many_zeros = 0

    for obs in ts["observations"]:
        if obs["value"] != '.':
            obs["value"] = float(obs["value"])
            curr.append(obs)
        else:
            sub_lists.append(curr)
            curr = []
            contains_na = True
    sub_lists.append(curr)

    freq, freq_short = find_frequency(ts["frequency"])

    """
    if contains_na:
        new_jsons = []
    else:
        new_ts = {
            "frequency": freq_short,
            "observations": curr,
            }
        new_jsons = [new_ts]
    return new_jsons, contains_na, freq, too_many_zeros, not_enough_distinct 

    """
    # Check if the time series are valid
    # THis is kinda wrong because we need to use l["value"] when checking for stuff.
    sub_lists = [l for l in sub_lists if len(l) > min_lengths[freq]]

    valid = []
    for l in sub_lists:
        if remove_zero:
            if np.count_nonzero(np.array(l)) < remove_zero_treshold * len(l):
                vals = np.array(l)
                if len(np.unique(vals)) > num_distinct:
                    valid.append(l)
                #else:
                #    not_enough_distinct = True
            #else:
            #    too_many_zeros = True
        else:
            vals = np.array(l)
            if len(np.unique(vals)) > num_distinct:
                valid.append(l)
            #else:
            #    not_enough_distinct = True

    new_jsons = [] 
    for l in sub_lists:
        new_ts = {
            "frequency" : freq_short,
            "observations" : list(l),
        }
        new_jsons.append(new_ts)

    return new_jsons, contains_na, freq, too_many_zeros, not_enough_distinct

#def preprocess_FRED(account_url:str, container_name:str, dates=False)->None:
def preprocess_FRED(cfg_preprocess)->None:
    """
    Preprocess the FRED dataset.
    """
    num_files_written = 0
    num_ts = 0
    num_contains_na = 0
    num_ts_processed = 0
    num_not_enough_distinct = 0
    num_too_many_zeros = 0
    all_frequencies:Dict= {}
    list_json = []
    
    dir_names = glob.glob(os.path.join(cfg_preprocess.raw_data_folder, "*")) 
    for dir_name in dir_names:
        logger.info(f"Preprocessing directory: {dir_name}")
        json_fnames = glob.glob(os.path.join(dir_name, "*"))
        for i, json_fname in enumerate(json_fnames):
            with open(json_fname) as json_file:
                json_data = json.load(json_file)
                json_file.close()
            for ts in json_data:
        
                try:
                    processed_jsons, contains_na, freq, too_many_zeros, not_enough_distinct = process_ts(
                        ts, 
                        cfg_preprocess.min_lengths,
                        remove_zero=cfg_preprocess.remove_zero,
                        remove_zero_treshold=cfg_preprocess.zero_treshold,
                        num_distinct=cfg_preprocess.num_distinct,
                        )
                    num_ts += len(processed_jsons)
                    num_contains_na += contains_na
                    #num_not_enough_distinct += not_enough_distinct
                    #num_too_many_zeros += too_many_zeros
                    num_ts_processed += 1
                    if not contains_na:
                        if freq in all_frequencies.keys():
                            all_frequencies[freq] += len(processed_jsons)
                        else:
                            all_frequencies[freq] = len(processed_jsons)

                    list_json.extend(processed_jsons)

                    if len(list_json) > cfg_preprocess.samples_per_json:
                        filename = f"processed_{num_files_written:>06}.json"
                        if num_files_written % cfg_preprocess.files_per_folder == 0:
                            curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
                            os.makedirs(os.path.join(cfg_preprocess.path.FRED, curr_dir), exist_ok=True)
                        with open(os.path.join(*[cfg_preprocess.path.FRED, curr_dir, filename]), "w") as fp:
                            json.dump(list_json, fp, sort_keys=True, indent=4, separators=(",", ": "))
                            fp.close()

                        num_files_written += 1
                        list_json = []
                except Exception as e:
                    logger.warning(e)

        logger.info(f"Currently have {num_ts} time series")
        logger.info(f"Of the {num_ts_processed} time series processed, {num_contains_na/num_ts_processed*100:.2f}% contains missing values.")


    filename = f"processed_{num_files_written:>06}.json"
    curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
    os.makedirs(os.path.join(cfg_preprocess.path.FRED, curr_dir), exist_ok=True)
    with open(os.path.join(*[cfg_preprocess.path.FRED, curr_dir, filename]), "w") as fp:
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
    """
    for i, bname in enumerate(raw_blob_names):
        print(f"Name of blob: {bname}")
        blob_client = BlobClient(account_url=cfg_preprocess.account_url, container_name=cfg_preprocess.container_name, blob_name=bname)
        blob_json = json.loads(blob_client.download_blob().readall())
        
        for ts in blob_json:
            try:
                processed_jsons, contains_na, freq, too_many_zeros, not_enough_distinct = process_ts(
                    ts, 
                    cfg_preprocess.min_lengths,
                    remove_zero=cfg_preprocess.remove_zero,
                    remove_zero_treshold=cfg_preprocess.zero_treshold,
                    num_distinct=cfg_preprocess.num_distinct,
                    )
                num_ts += len(processed_jsons)
                num_contains_na += contains_na
                num_not_enough_distinct += not_enough_distinct
                num_too_many_zeros += too_many_zeros
                num_ts_processed += 1
                if freq in all_frequencies.keys():
                    all_frequencies[freq] += 1
                else:
                    all_frequencies[freq] = 1

                #if processed_jsons.size():
                #   list_jsons.extend(processed_jsons)
                list_json.extend(processed_jsons)

                if len(list_json) > cfg_preprocess.samples_per_json:
                    filename = f"processed_{num_files_written:>06}.json"
                    if num_files_written % cfg_preprocess.files_per_folder == 0:
                        curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
                        os.makedirs(os.path.join(cfg_preprocess.path.FRED, curr_dir), exist_ok=True)
                    with open(os.path.join(*[cfg_preprocess.path.FRED, curr_dir, filename]), "w") as fp:
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

        break

    filename = f"processed_{num_files_written:>06}.json"
    if num_files_written % cfg_preprocess.files_per_folder == 0:
        curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
        os.makedirs(os.path.join(cfg_preprocess.path.FRED, curr_dir), exist_ok=True)
    with open(os.path.join(*[cfg_preprocess.path.FRED, curr_dir, filename]), "w") as fp:
        json.dump(list_json, fp, sort_keys=True, indent=4, separators=(",", ": "))
        fp.close()

    logger.info(f"Processed {num_ts_processed} files")
    logger.info(f"Currently have {num_ts} time series")
    logger.info(f"Of the {num_ts_processed} time series processed, {num_contains_na/num_ts_processed*100:.2f}% contains missing values.")
    logger.info(f"Of the {num_ts_processed} time sereis processed, {num_too_many_zeros/num_ts_processed*100:.2f}% had too many zeros.")
    logger.info(f"Of the {num_ts_processed} time series processed, {num_not_enough_distinct/num_ts_processed*100:.2f}% did not have enough distinct values")
    logger.info("Proportion of frequencies: ")
    tot_freq = sum(all_frequencies.values())
    for k, v in all_frequencies.items():
        logger.info(f"{k} : {v/tot_freq*100:.2f}")
    logger.info("Done preprocessing FRED.")
    """


if __name__ == "__main__":
    preprocess_FRED(account_url="https://tsdatasets.blob.core.windows.net/", container_name="fred")
