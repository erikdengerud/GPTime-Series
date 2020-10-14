import torch
from torch.utils.data import Dataset

import logging
import glob
import json
import os
import numpy as np
import time
from typing import Dict, List, Tuple
import sys

sys.path.append("")

from GPTime.config import cfg
from GPTime.utils.scaling import MASEscale

logger = logging.getLogger(__name__)

class TSDataset(Dataset):
    """
    Time series dataset.
    TODO:
    * Horizontal batch size. This is not easy wrt. scaling!
    """
    def __init__(
        self, 
        memory:int, 
        dataset_paths:Dict[str, str], 
        frequencies:Dict[str, bool],
        min_lengths:Dict[str, int],
        cutoff_date:str=None,  
        convolutions:bool=False,
        horizontal_batch_size:int=1,
        ) -> None:
        super(TSDataset, self).__init__()
        self.memory = memory
        self.convolutions = convolutions
        self.horizontal_batch_size = horizontal_batch_size
        self.min_lengths = min_lengths

        if cutoff_date is not None:
            cutoff_date_date = time.strptime(cutoff_date, "%Y-%m-%d")

        # Read data into memory
        all_ts = []
        for dataset_path in dataset_paths.values():
            dirs = glob.glob(os.path.join(dataset_path, "*"))
            for d in dirs:
                logger.info(f"Loading dir: {d}")
                fnames = glob.glob(os.path.join(d, "*"))
                for fname in fnames:
                    with open(fname, "r") as fp:
                        json_list = json.load(fp)
                        fp.close()
                    for ts in json_list:
                        if frequencies[ts["frequency"]]:
                            if cutoff_date is not None:
                                vals = []
                                for obs in ts["observations"]:
                                    if time.strptime(obs["date"], "%Y-%m-%d") < cutoff_date_date:
                                        vals.append(float(obs["value"])) 
                                vals = np.array(vals)
                            else:
                                vals = np.array([float(obs["value"]) for obs in ts["observations"]])
                            new_ts = {
                                "frequency": ts["frequency"],
                                "values" : np.array([float(obs["value"]) for obs in ts["observations"]])
                            }
                            all_ts.append(new_ts)
        self.all_ts = all_ts
        logger.info(f"Total of {len(all_ts)} time series in memory.")

    def __len__(self):
        return len(self.all_ts)

    def __getitem__(self, idx) -> Tuple:
        if not isinstance(idx, list):
            idx = [idx]
        #logger.info(idx)
        #logger.info(type(idx))
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        #ids = list(idx)
        ids_ts = [self.all_ts[i] for i in idx]
        frequencies = [obs["frequency"] for obs in ids_ts]
        values = [obs["values"] for obs in ids_ts]
        #logger.info(values)
        #for val in values:
        #    logger.info(val.dtype)

        # Sample ts
        samples, labels = self.sample_ts(values, frequencies)

        samples_tensor = torch.from_numpy(samples)
        labels_tensor = torch.from_numpy(labels)

        if self.convolutions:
            samples_tensor = samples_tensor.unsqueeze(1)

        return samples_tensor, labels_tensor, frequencies


    def sample_ts(self, ts_list:List[np.array], frequency_list:List[str]) -> Tuple[np.array, np.array]:
        """
        Create a sample from a time series.

        Sampling algorithm:
            1. Sample length in range min_length to length of model memory. Frequency dependent.
            2. Sample end index in range 0 length of series - 1.
            3. Create the sample by choosing the observations in range [max(0, end_index - length), end_index]
            4. Use the end_index + 1 entry of the time series as label.
            5. Scale the sample using the scaling in MASE.
            6. Scale label using the same scale as for the sample. 
        """
        samples = []
        labels = []
        for ts, freq in zip(ts_list, frequency_list):
            min_length = self.min_lengths[freq]
            max_length = max(self.memory + self.horizontal_batch_size, min_length+self.horizontal_batch_size)

            length = np.random.randint(min_length, max_length)
            end_index = np.random.randint(min_length, len(ts)-1)
            logger.info(f"length:{length}")
            logger.info(f"end_index:{end_index}")

            sample = ts[max(0, end_index - length): end_index]
            label = ts[end_index]

            scale = MASEscale(sample, freq)
            sample_scaled = sample / scale
            label_scaled = label / scale
            
            logger.info(f"length of sample: {len(sample_scaled)}")
            logger.info(f"pad length: {self.memory - len(sample_scaled)}")
            sample_scaled_pad = np.pad(sample_scaled, (self.memory - len(sample_scaled), 0), mode="constant", constant_values=0)
            
            assert len(sample_scaled) == self.memory, f"Sample is not as long as memory. Length sample: {len(sample_scaled)}, memory:{self.memory}"

            samples.append(sample_scaled_pad)
            labels.append(label_scaled)

        samples = np.vstack(samples)
        labels = np.vstack(labels)

        return samples, labels


if __name__=="__main__":
    ds = TSDataset(
        memory=10,
        convolutions=False,
        **cfg.dataset.dataset_params)
    logger.debug(ds.__len__())
    #x, y, f = ds.__getitem__([1,3,100])
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset=ds, batch_size=2, shuffle=True, num_workers=4)
    for i, data in enumerate(dl):
        x, y, f = data
        logger.info(f"Batch {i+1}")
        break
    #logger.debug(x.shape)
    #logger.info(x)