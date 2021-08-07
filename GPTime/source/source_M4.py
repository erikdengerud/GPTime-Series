import logging
import os
import sys
import pandas as pd
sys.path.append("")

from GPTime.config import cfg

logger = logging.getLogger(__name__)

def download_file(dpath, store_path)->None:
    """ Downloads file and stores in store path with derived filename. """
    try:
        logger.info(f"Downloading {dpath}")
        df = pd.read_csv(dpath, index_col=0)
        out_fname = dpath.split("/")[-1]
        out_fname = os.path.join(store_path, out_fname)
        df.to_csv(out_fname)
    except Exception as e:
        logger.info(f"Failed to download file from {dpath}")
        logger.info(e)


def source_M4(train_download_paths, test_download_paths, meta_download_path, train_store_path, test_store_path, meta_store_path)->None:
    """ Download M4 data from the M4 github repo. """
    # Create directories
    os.makedirs(train_store_path, exist_ok=True)
    os.makedirs(test_store_path, exist_ok=True)
    os.makedirs(meta_store_path, exist_ok=True)
    # Train
    for dpath in train_download_paths:
        download_file(dpath, train_store_path)
    # Test
    for dpath in test_download_paths:
        download_file(dpath, test_store_path)
    # Meta 
    download_file(meta_download_path, meta_store_path)
