import torch
from torch.utils.data import Dataset

import logging
import glob
import json
import os
import numpy as np

import sys

sys.path.append("")

from GPTime.config import cfg

logger = logging.getLogger(__name__)

class TSDataset(Dataset):
    """
    Time series dataset.
    TODO:
        * Horizontal batching for TCN.
    """
    def __init__(self, memory:int, dataset_path:str, convolutions:bool=False):
        super(TSDataset, self).__init__()
        self.memory = memory
        self.convolutions = convolutions
        # read data into memory
        ts = []
        dir_names = []
        paths = glob.glob(dataset_path + "*") if dataset_path[-1] == "/" else glob.glob(dataset_path + "/*")
        if len(paths) == 0:
            logger.warning("No directories in dataset path.")
        for path in paths:
            dir_names.append(path)
            if not os.path.isdir(path):
                logger.warning("Not all paths in dataset folder are folders.")
        logger.debug(f"Number of directories: {len(dir_names)}")

        for dir_name in dir_names[0:1]:
            fnames = glob.glob(path + "/*")
            for fname in fnames:
                if not os.path.isfile(fname):
                    logger.warning("File path is not a file.")
                with open(fname, "r") as f:
                    in_file = json.load(f)
                    ts.append(in_file)
                    f.close()
            #logger.debug(f"Number of files in directory {dir_name} : {len(fnames)}")
            logger.debug(f"Number of tim series in dataset: {len(ts)}")
        self.ts = ts

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx):
        ids = list(idx)
        ids_ts =  [self.ts[i] for i in ids]
        frequencies = [obs["frequency"] for obs in ids_ts]
        values = [obs["values"] for obs in ids_ts]
        
        # call sample ts here

        values_tensors = torch.from_numpy(np.array(values))
        if self.convolutions:
            values_tensors = values_tensors.unsqueeze(1)
        return values_tensors

    @staticmethod
    def sample_ts(ts, frequency):
        # sample, pad
        pass

if __name__=="__main__":
    ds = TSDataset(cfg.train.model.memory, cfg.preprocess.path, convolutions=True)
    logger.debug(ds.__len__())
    x = ds.__getitem__([1,3,100])
    logger.debug(x.shape)