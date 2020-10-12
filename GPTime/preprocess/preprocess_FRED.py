"""
Preprocess: Read from staging DB/files, use asserts to validate key staged data 
properties, engineer features, transform, and save to new DB table(s) or files.
"""

"""
Preprocess: Read the dummy table/file written by source. Don’t do anything with it. 
Write dummy preprocessed data for predict.
"""

import numpy as np
import glob
import logging
import os
import json
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import sys

sys.path.append("")
from GPTime.config import cfg

logger = logging.getLogger(__name__)


def preprocess_FRED(account_url:str, container_name:str)->None:
    """
    Preprocess the FRED dataset.
    """

    # Azure clients
    container_client = ContainerClient(account_url=account_url, container_name=container_name)
    all_blobs = container_client.list_blobs()
    raw_blobs = [b["name"] for b in all_blobs if "raw" in b["name"]]
    
    for b in raw_blobs:
        bname = b["name"]
        print(f"Name of blob: {bname}")
        blob_client = BlobClient(account_url=account_url, container_name=container_name, blob_name=bname)
        download_stream = blob_client.download_blob()
            my_blob.write(download_stream.readall())
        with open(bname, "wb") as my_blob:
            download_stream = blob_client.download_blob()
            my_blob.write(download_stream.readall())
            my_blob.close()
        break
    with open(bname, "rb") as fp:
        blob_json = json.load(fp)
        fp.close()
    print(f"Number of samples in file: {len(blob_json)}\n")
    print("Example of metadata for a time series:\n")
    for key, value in blob_json[1].items():
        print(f"{key}: {value}")
    """
    # for each blob

        # for each ts in blob

            # Check validity

            # save

if __name__ == "__main__":
    preprocess_FRED(account_url="https://tsdatasets.blob.core.windows.net/", container_name="fred")