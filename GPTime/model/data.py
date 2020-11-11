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

Scaler = getattr(importlib.import_module(cfg.train.scaler_module), cfg.train.scaler_name)

logger = logging.getLogger(__name__)

class TSDataset(Dataset):
    """
    Time series dataset.
    """
    def __init__(
        self, 
        memory:int, 
        dataset_paths:Dict[str, str], 
        frequencies:Dict[str, bool],
        min_lengths:Dict[str, int],
        cutoff_date:str=None,  
        convolutions:bool=False,
        ) -> None:
        super(TSDataset, self).__init__()
        assert memory >= max(min_lengths.values()), "Increase model memory, it is shorter than max of min lengths."
        self.memory = memory
        self.convolutions = convolutions
        self.min_lengths = min_lengths
        self.cutoff_date = cutoff_date
        if cutoff_date is not None:
            self.cutoff_date_date = time.strptime(cutoff_date, "%Y-%m-%d")

        # Read data into memory
        all_ts = []
        for dataset_path in dataset_paths.values():
            dirs = glob.glob(os.path.join(dataset_path, "*"))
            for d in dirs:
                logger.info(f"Loading dir: {d}")
                fnames = glob.glob(os.path.join(d, "*"))
                for fname in fnames:
                    with open(fname, "r") as fp:
                        ts_list = json.load(fp)
                        fp.close()
                    for ts in ts_list:
                        if frequencies[ts["frequency"]]:
                            prepared_ts = self.prepare_ts(ts)
                            if prepared_ts["values"].size:
                                all_ts.append(prepared_ts)
        self.all_ts = all_ts
        logger.info(f"Total of {len(all_ts)} time series in memory.")

    def __len__(self):
        return len(self.all_ts)

    def __getitem__(self, idx) -> Tuple:
        id_ts = self.all_ts[idx]
        frequency = id_ts["frequency"]
        values = id_ts["values"]

        sample, label, mask = self.sample_ts(values, frequency)

        sample_tensor = torch.from_numpy(sample)
        label_tensor = torch.from_numpy(label)
        mask_tensor = torch.from_numpy(mask)

        if self.convolutions:
            sample_tensor = sample_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.unsqueeze(0)

        return sample_tensor, label_tensor, mask_tensor, frequency

    def sample_ts(self, ts:np.array, freq:str) -> Tuple[np.array, np.array, np.array]:
        """Sampling a training sample a time series.

        Sampling algorithm:
            1. Sample sample_length in range min_length to length of model memory. Frequency dependent.
            2. Sample end index in range sample_length to length of time series.
            3. Create the sample by choosing the observations in range [(end_index - sample_length), end_index]
            4. Use the end_index entry of the time series as label.

        Args:
            ts (np.array): A time series in the form of an array.
            freq (str): The frequency of the time series. One capitalized letter.

        Returns:
            Tuple[np.array, np.array, np.arrray]: The sample, label and mask.
        """
        min_length = self.min_lengths[freq]
        #max_length = min(self.memory, len(ts))
        max_length = min(self.memory, len(ts)) + 1
        assert min_length < max_length, f"min_length ({min_length}) longer than max_length ({max_length})"

        length = np.random.randint(min_length, max_length) - 1
        assert length < len(ts), f"length ({length}) longer than len(ts) ({len(ts)})"
        end_index = np.random.randint(length, len(ts))

        sample = ts[(end_index - length) : end_index]
        label = np.array(ts[end_index])

        sample_mask = np.zeros(self.memory)
        sample_mask[-len(sample):] = 1.0

        sample = np.pad(sample, (self.memory - len(sample), 0), mode="constant", constant_values=0)

        assert len(sample) == self.memory, f"Sample is not as long as memory. Length sample: {len(sample)}, memory:{self.memory}"
        assert len(sample_mask) == self.memory, f"Mask is not as long as memory. Length mask: {len(sample_mask)}, memory:{self.memory}"
    
        return sample, label, sample_mask

    def prepare_ts(self, ts:Dict) -> Dict:
        """Prepare a preprocessed time series for training. 
        Use only values before a cutoff date. 
        Storing observations as floats in a numpy array. 
        Scaling the time series using the full time series. Required to stabilize training with some Scalers. The best would be to scale each sample by itself.

        Args:
            ts (Dict): A time series with observations and metadata. Needs frequency and observations.

        Returns:
            Dict: The same time series with values from before the cutoff date as floats in a numpy array and the frequency as a string.
        """
        if self.cutoff_date is not None:
            vals = []
            for obs in ts["observations"]:
                if time.strptime(obs["date"], "%Y-%m-%d") < self.cutoff_date_date:
                    vals.append(float(obs["value"])) 
            vals = np.array(vals)
        else:
            vals = np.array([float(obs["value"]) for obs in ts["observations"]])
        
        freq_str = ts["frequency"]
        freq_int = cfg.dataset.scaling.periods[freq_str]
        if Scaler.__name__ == "MASEScaler":
            values_scaled = Scaler().fit_transform(vals, freq=freq_int)
        else:
            values_scaled = Scaler().fit_transform(vals.reshape(1,-1).T).flatten().T

        if len(values_scaled) < self.min_lengths[ts["frequency"]]:
            values_scaled = np.array([])

        prepared_ts = {
            "frequency": ts["frequency"],
            "values" : values_scaled
        }

        return prepared_ts
        

class DeterministicTSDataSet(Dataset):
    """
    Deterministic version of the TSDataset for validation and testing.
    """
    def __init__(
        self, 
        dataset,
        num_tests_per_ts:int,
        )->None:
        super(DeterministicTSDataSet, self).__init__()
        self.memory = dataset.dataset.memory
        self.convolutions = dataset.dataset.convolutions
        self.min_lengths = dataset.dataset.min_lengths
        all_ts = []
        for ind in dataset.indices:
            all_ts.append(dataset.dataset.all_ts[ind])
        self.all_ts = all_ts
        self.num_tests_per_ts = num_tests_per_ts
        
        self.all_samples, self.all_labels, self.all_frequencies = self.create_deterministic_samples()
        logging.info("Created deterministic samples.")
        
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        #id_ts = self.all_samples[idx]
        #frequency = id_ts["frequency"]
        #values = id_ts["values"]
        sample = self.all_samples[idx]
        label = self.all_labels[idx]
        frequency = self.all_frequencies[idx]

        sample_tensor = torch.from_numpy(sample)
        label_tensor = torch.from_numpy(label)

        if self.convolutions:
            sample_tensor = sample_tensor.unsqueeze(0)

        return sample_tensor, label_tensor, frequency

    def create_deterministic_samples(self)-> Tuple[np.array, np.array, np.array]:
        all_samples = []
        all_labels = []
        all_frequencies = []
        sample_indexes = [i // self.num_tests_per_ts for i in range(len(self.all_ts)*self.num_tests_per_ts)]
        for si in sample_indexes:
            freq = self.all_ts[si]["frequency"]
            values = self.all_ts[si]["values"]
            sample, label = self.sample_ts(values, freq)
            all_samples.append(sample)
            all_labels.append(label)
            all_frequencies.append(freq)
        return all_samples, all_labels, all_frequencies

    def sample_ts(self, ts:np.array, freq:str) -> Tuple[np.array, np.array]:
        min_length = self.min_lengths[freq]
        max_length = self.memory
        length = np.random.randint(min_length, max_length)
        end_index = np.random.randint(min_length, len(ts))
        sample = ts[max(0, end_index - length) : end_index]
        label = np.array(ts[end_index])
        scale = MASEscale(sample, freq)
        sample_scaled = sample / scale
        label_scaled = label / scale
        sample_scaled_pad = np.pad(sample_scaled, (self.memory-len(sample_scaled), 0), mode="constant", constant_values=0)
        assert len(sample_scaled_pad) == self.memory, f"Sample is not as long as memory. Length sample: {len(sample_scaled_pad)}, memory:{self.memory}"
        return np.asarray(sample_scaled_pad), np.asarray(label_scaled)


class DummyDataset(Dataset):
    """
    Dummy dataset for debugging. Using the same structure as the time series dataset.
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
        super(DummyDataset, self).__init__()
        assert memory > max(min_lengths.values()), "Increase model memory, it is shorter than max of min lengths."
        self.memory = memory
        self.convolutions = convolutions
        self.min_lengths = min_lengths

        # Read data into memory
        all_ts = []
        for i in range(100):
            l = np.random.randint(100, 200)
            vals = np.array([i%2 for i in range(l)])
            new_ts = {
                "frequency": np.random.choice(["Y", "Q", "M", "W", "D", "H", "O"]),
                "values" : vals
            }
            all_ts.append(new_ts)
        self.all_ts = all_ts
        logger.info(f"Total of {len(all_ts)} time series in memory.")

    def __len__(self):
        return len(self.all_ts)

    def __getitem__(self, idx) -> Tuple:
        id_ts = self.all_ts[idx]
        frequency = id_ts["frequency"]
        values = id_ts["values"]

        # Sample ts
        sample, label = self.sample_ts(values, frequency)

        sample_tensor = torch.from_numpy(sample).double()
        label_tensor = torch.from_numpy(label).double()

        if self.convolutions:
            sample_tensor = sample_tensor.unsqueeze(0)

        return sample_tensor, label_tensor, frequency

    def sample_ts(self, ts:np.array, freq:str) -> Tuple[np.array, np.array]:

        min_length = self.min_lengths[freq]
        max_length = self.memory

        length = np.random.randint(min_length, max_length)
        end_index = np.random.randint(min_length, len(ts))

        sample = ts[max(0, end_index - length) : end_index]
        label = np.array(ts[end_index])
        
        sample_pad = np.pad(sample, (self.memory-len(sample), 0), mode="constant", constant_values=0)
        assert len(sample_pad) == self.memory, f"Sample is not as long as memory. Length sample: {len(sample_pad)}, memory:{self.memory}"

        return np.asarray(sample_pad), np.asarray(label)

class MonteroMansoHyndmanDS(Dataset):
    """
    Dataset on the form of Monetero-Manso and Hyndman. Last 12:1 observations of each ts
    from M4 as samples. Last obs as label. Scaled by MASE.
    """
    def __init__(
        self, 
        memory:int, 
        dataset_paths:Dict[str, str], 
        frequencies:Dict[str, bool],
        min_lengths:Dict[str, int],
        cutoff_date:str=None,  
        convolutions:bool=False,
        ) -> None:
        super(MonteroMansoHyndmanDS, self).__init__()
        assert memory >= max(min_lengths.values()), "Increase model memory, it is shorter than max of min lengths."
        self.memory = memory
        self.convolutions = convolutions
        self.min_lengths = min_lengths
        self.cutoff_date = cutoff_date
        if cutoff_date is not None:
            self.cutoff_date_date = time.strptime(cutoff_date, "%Y-%m-%d")

        #logger.debug(dataset_paths)
        # Read data into memory
        all_ts = []
        for dataset_path in dataset_paths.values():
            dirs = glob.glob(os.path.join(dataset_path, "*"))
            #logger.debug(dirs)
            for d in dirs:
                logger.info(f"Loading dir: {d}")
                fnames = glob.glob(os.path.join(d, "*"))
                for fname in fnames:
                    with open(fname, "r") as fp:
                        ts_list = json.load(fp)
                        fp.close()
                    for ts in ts_list:
                        if frequencies[ts["frequency"]]:
                            sample, label = self.prepare_ts(ts)
                            if sample.size:
                                all_ts.append((sample, label))
        self.all_ts = all_ts
        logger.info(f"Total of {len(all_ts)} time series in memory.")

    def __len__(self):
        return len(self.all_ts)

    def __getitem__(self, idx) -> Tuple:
        sample, label = self.all_ts[idx]
        sample_tensor = torch.from_numpy(np.asarray(sample))
        label_tensor = torch.from_numpy(np.asarray(label))
        if self.convolutions:
            sample_tensor = sample_tensor.unsqueeze(0)
        return sample_tensor, label_tensor, 0

    def prepare_ts(self, ts:Dict) -> Tuple:
        freq_str = ts["frequency"]
        freq_int = cfg.dataset.scaling.periods[freq_str]
        values = np.array([float(obs["value"]) for obs in ts["observations"]])
        #sample = values[-(self.memory + 1):-1]
        #label = values[-1]
        if Scaler.__name__ == "MASEScaler":
            #scaler = Scaler().fit(sample, freq=freq_int)
            values_scaled = Scaler().fit_transform(values, freq=freq_int)
        else:
            scaler = Scaler().fit(values)

        sample_scaled = values_scaled[-(self.memory + 1):-1]
        label_scaled = values_scaled[-1]
        #sample_scaled = scaler.transform(sample)
        #label_scaled = scaler.transform(label)
    
        return sample_scaled, label_scaled

class MonteroMansoHyndmanSimpleDS(Dataset):
    """
    Dataset on the form of Monetero-Manso and Hyndman. Last 12:1 observations of each ts
    from M4 as samples. Last obs as label. Scaled by MASE.
    """
    def __init__(
        self, 
        arr:np.array,
        memory:int, 
        dataset_paths:Dict[str, str], 
        frequencies:Dict[str, bool],
        min_lengths:Dict[str, int],
        cutoff_date:str=None,  
        convolutions:bool=False,
        ) -> None:
        super(MonteroMansoHyndmanSimpleDS, self).__init__()
        logger.info("Simple dataset.")
        #arr = np.load(dataset_paths["M4_global"])

        self.X = torch.from_numpy(arr)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :-1], self.X[idx, -1], 0

class FREDSimpleDS(Dataset):
    """
    Dataset on the form of Monetero-Manso and Hyndman. Last 12:1 observations of each ts
    from M4 as samples. Last obs as label. Scaled by MASE.
    """
    def __init__(
        self, 
        memory:int, 
        dataset_paths:Dict[str, str], 
        frequencies:Dict[str, bool],
        min_lengths:Dict[str, int],
        cutoff_date:str=None,  
        convolutions:bool=False,
        ) -> None:
        super(FREDSimpleDS, self).__init__()
        logger.info("Simple FRED dataset.")
        arr = np.load("FRED_small.npy")

        self.X = torch.from_numpy(arr)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :-1], self.X[idx, -1], 0, 0


if __name__=="__main__":
    ds = FREDSimpleDS(
        memory=100,
        convolutions=False,
        **cfg.dataset.dataset_params)
    logger.debug(ds.__len__())

    from torch.utils.data import DataLoader
    dl = DataLoader(dataset=ds, batch_size=1, shuffle=True, num_workers=4)
    for i, data in enumerate(dl):
        sample, label, mask = data[0].numpy(), data[1].numpy(), data[2].numpy()
        logger.info(f"Batch {i+1}")
        logger.debug(sample.shape)
        logger.debug(label.shape)
        logger.debug(mask.shape)
        logger.debug(sample)
        logger.debug(label.item())
        logger.debug(mask)
        logger.debug(np.sum(sample))


        break

    """
    ds = MonteroMansoHyndmanSimpleDS(
        memory=12,
        convolutions=False,
        **cfg.dataset.dataset_params)
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset=ds, batch_size=16, shuffle=True, num_workers=4)
    for i, data in enumerate(dl):
        x, y, f = data
        logger.info(f"Batch {i+1}")
        logger.debug(x.shape)
        logger.debug(y.shape)
        logger.debug(x)
        logger.debug(y)
        #output = model(x)
        #logger.info(output.shape)
        break
    """
    """
    ds = DummyDataset(
        memory=100,
        convolutions=False,
        **cfg.dataset.dataset_params)
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset=ds, batch_size=1, shuffle=True, num_workers=4)
    for i, data in enumerate(dl):
        x, y, f = data
        logger.info(f"Batch {i+1}")
        logger.debug(x.shape)
        logger.debug(y.shape)
        logger.debug(x)
        logger.debug(y)
        #output = model(x)
        #logger.info(output.shape)
        break
    """
    """
    ds = TSDataset(
        memory=100,
        convolutions=True,
        **cfg.dataset.dataset_params)
    logger.debug(ds.__len__())
    #x, y, f = ds.__getitem__([1,3,100])
    from torch.utils.data import random_split
    train_length = int(ds.__len__() * cfg.train.train_set_size)
    val_length = int(ds.__len__() * cfg.train.val_set_size)
    test_length = ds.__len__() - train_length - val_length
    train_ds, val_ds, test_ds = random_split(
        dataset=ds, 
        lengths=[train_length, val_length, test_length],
        generator=torch.torch.Generator()
        )
    logger.info(train_ds.__len__())
    logger.info(val_ds.__len__())
    logger.info(test_ds.__len__())

    ds_val = DeterministicTSDataSet(val_ds, num_tests_per_ts=3)
    logger.info(ds_val.__len__())
    x, y, f = ds_val.__getitem__(1)
    logger.debug(x.shape)
    logger.debug(y.shape)
    logger.info(x)
    logger.info(y)
    logger.info(f)
    x1, y, f = ds_val.__getitem__(1)
    x2, y, f = ds_val.__getitem__(1)
    x3, y, f = ds_val.__getitem__(1)
    logger.info(x1)
    logger.info(x2)
    logger.info(x3)
    """

    """
    from GPTime.networks.mlp import MLP
    from GPTime.networks.tcn import TCN
    #model = MLP(in_features=100, out_features=1, num_layers=5, n_hidden=32, bias=True).double()
    model = TCN(in_channels=1, channels=[8,8,8,1], kernel_size=2).double()
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset=ds, batch_size=1024, shuffle=True, num_workers=4)
    for i, data in enumerate(dl):
        x, y, f = data
        logger.info(f"Batch {i+1}")
        logger.debug(x.shape)
        logger.debug(y.shape)
        output = model(x)
        logger.info(output.shape)
    """
    """
    logger.debug(x.shape)
    logger.debug(y.shape)
    logger.info(x)
    logger.info(y)
    logger.info(f)
    """
