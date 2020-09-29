"""
Source: Pipe data from multiple original sources into a staging DB (e.g., for simple 
applications, sqlite), separate files (e.g., images for training a classifier), or 
denormalized tables saved as pickle, Parquet, feather, or other formats.
"""

"""
Source: For each data store from which data is sourced, query a small sample of the 
data (not saved anywhere). Instead, write a dummy data table/file to the designated 
staging DB/bucket/directory for preprocess to pick up.
"""
import sys
sys.path.append("")

from GPTime.source.source_FRED import source_FRED

from typing import Dict
import logging
from GPTime.config import cfg

logger = logging.getLogger(__name__)


def source(credentials, small_sample:bool=False)->None:
    """
    Sourcing raw data from data sources.
    """

    # Fred
    source_FRED(credentials=credentials.FRED, small_sample=small_sample)

    
"""
if __name__ == "__main__":
    import yaml
    import sys
    sys.path.append("")
    sys.path.append("../..")
    from box import Box
    #import matplotlib.pyplot as plt  

    with open("GPTime/credentials.yml", "r") as ymlfile:
        credentials = Box(yaml.safe_load(ymlfile))

    source(credentials, small_sample=True)
"""