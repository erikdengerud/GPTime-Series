import importlib
import logging
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import pandas as pd
import time
import sys

sys.path.append("")




from GPTime.config import cfg
from GPTime.model.data import DeterministicTSDataSet

from GPTime.model.data import TSDataset_iter

logger = logging.getLogger(__name__)

Model = getattr(importlib.import_module(cfg.train.model_module), cfg.train.model_name)
Criterion = getattr(
    importlib.import_module(cfg.train.criterion_module), cfg.train.criterion_name
)
Optimizer = getattr(
    importlib.import_module(cfg.train.optimizer_module), cfg.train.optimizer_name
)
Dataset = getattr(
    importlib.import_module(cfg.dataset.dataset_module), cfg.dataset.dataset_name
)
DataLoader = getattr(
    importlib.import_module(cfg.train.dataloader_module), cfg.train.dataloader_name
)

def mase(forecast, insample, outsample, frequency):
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))

def train():
    #np.random.seed(1729)
    #torch.manual_seed(1729)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if Model.__name__ == "MLP":
        model_params = cfg.train.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = cfg.train.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = cfg.train.model_params_tcn
    else:
        logger.warning("Unknown model name.")
    model = Model(**model_params).double()
    model.to(device)
    logger.info(
        f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    #criterion = Criterion(**cfg.train.criterion_params)
    criterion = Criterion
    optimizer = Optimizer(model.parameters(), **cfg.train.optimizer_params)
    writer = SummaryWriter(log_dir=cfg.train.tensorboard_log_dir)

    # Dataset
    ds = Dataset(
        memory=model.memory,
        convolutions=True if Model.__name__=="TCN" else False,
        **cfg.dataset.dataset_params
        )

    # Proportion of ds to use
    assert cfg.dataset.proportion <= 1 and cfg.dataset.proportion > 0, "Proportion of dataset not between 0 and 1."
    proportion_length = int(ds.__len__() * cfg.dataset.proportion)
    ds_use, _ = random_split(dataset=ds, lengths=[proportion_length, ds.__len__() - proportion_length])

    train_length = int(ds_use.__len__() * cfg.train.train_set_size)
    val_length = int(ds_use.__len__() * cfg.train.val_set_size)
    test_length = ds_use.__len__() - train_length - val_length

    train_ds, val_ds, test_ds = random_split(
        dataset=ds_use, 
        lengths=[train_length, val_length, test_length],
        #generator=torch.torch.Generator()
    )
    logger.info(f"Using {cfg.dataset.proportion * 100}% of the available dataset.")
    logger.info(f"Using frequencies: {[freq for freq, true_false in cfg.dataset.dataset_params.frequencies.items() if true_false]}")
    logger.info(f"Train size: {train_ds.__len__()}, Val size: {val_ds.__len__()}, Test size: {test_ds.__len__()}")

    # Dataloader
    trainloader = DataLoader(train_ds, **cfg.train.dataloader_params)
    valloader = DataLoader(val_ds, **cfg.train.dataloader_params)
    testloader = DataLoader(train_ds, **cfg.train.dataloader_params)
    
    
    training_set = TSDataset_iter([ts["values"] for ts in ds.all_ts])
    iter_training_set = iter(training_set)

    lr_decay_step = cfg.train.max_epochs // 3
    if lr_decay_step == 0:
        lr_decay_step = 1

    running_loss = 0.0
    for i in range(1, cfg.train.max_epochs + 1):
        time_epoch = time.time()
        iteration_loss = 0.0
        model.train()
        data = next(iter_training_set) #data[0].to(device), data[1].to(device), data[2].to(device)
        inputs, labels, mask = torch.from_numpy(data[0]).to(device), torch.from_numpy(data[1]).to(device), torch.from_numpy(data[2]).to(device)
        #freq = [cfg.dataset.scaling.periods[f] for f in data[3]]
        
        optimizer.zero_grad()
        forecast = model(inputs, mask, _)
        loss = criterion(forecast, labels, mask, inputs, 12) # change
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        iteration_loss += loss.item()

        for param_group in optimizer.param_groups:
            param_group["lr"] = cfg.train.optimizer_params.lr * 0.5 ** (i//lr_decay_step)

        if i % cfg.train.log_freq == 0:
            logger.info(f"Epoch {i:>3}: {time.time()-time_epoch:.2f}s/iter, [log_freq_loss, iter_loss]: [{running_loss:.6f}, {iteration_loss:.6f}]")
            running_loss = 0.0

    # Build forecasts
    train_data, mask = training_set.last_insample_window()
    model.eval()
    with torch.no_grad():
        for i in range(18):
            sample = torch.from_numpy(train_data[:, -model.memory:])
            sample_mask = torch.from_numpy(mask[:,-model.memory:])
            out = model(sample, sample_mask, _).cpu().detach().numpy()
            train_data = np.hstack((train_data, out))
            mask = np.hstack((mask, np.ones((mask.shape[0], 1))))
        forecast = train_data[:, -18:]

    df_test_monthly = pd.read_csv(os.path.join(cfg.source.path.M4.raw_test, "Monthly-test.csv"))
    df_train_monthly = pd.read_csv(os.path.join(cfg.source.path.M4.raw_train, "Monthly-train.csv"))
    in_sample = np.array([ts[~pd.isnull(ts)] for ts in df_train_monthly.values], dtype=object)
    target = np.array([ts[~pd.isnull(ts)] for ts in df_test_monthly.values], dtype=object)
    assert forecast.shape == target.shape, f"forecast.shape: {forecast.shape}, target.shape: {target.shape}"
    mase_res = np.mean([mase(forecast=forecast[i], insample=in_sample[i], outsample=target[i], frequency=12) for i in range(len(forecast))])
    logger.debug(f"MASE nbeats way = {mase_res}")


    """
    early_stop_count = 0
    low_loss = np.inf
    for ep in range(1, cfg.train.max_epochs + 1):
        model.train()
        time_epoch = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, mask = data[0].to(device), data[1].to(device), data[2].to(device)
            freq = [cfg.dataset.scaling.periods[f] for f in data[3]]
            
            optimizer.zero_grad()
            forecast = model(inputs, mask, freq)
            loss = criterion(forecast, labels, mask, inputs, 12) # change
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                inputs, labels, mask = data[0].to(device), data[1].to(device), data[2].to(device)
                freq = [cfg.dataset.scaling.periods[f] for f in data[3]]
                outputs = model(inputs, mask, freq)
                loss = criterion(outputs, labels, mask, inputs, 12)
                running_val_loss += loss.item()
            if running_val_loss < low_loss:
                early_stop_count = 0
                low_loss = running_val_loss
            else:
                early_stop_count += 1
            if early_stop_count > cfg.train.early_stop_tenacity:
                print(f"Early stop after epoch {ep}.")
                break

        running_test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels, mask = data[0].to(device), data[1].to(device), data[2].to(device)
                freq = [cfg.dataset.scaling.periods[f] for f in data[3]]
                outputs = model(inputs, mask, freq)
                loss = criterion(outputs, labels, mask, inputs, 12)
                running_test_loss += loss.item()

        if ep % cfg.train.log_freq == 0:
            logger.info(f"Epoch {ep:>3}: {time.time()-time_epoch:.2f}s/ep, [train, val, test]: [{running_loss:.6f}, {running_val_loss:.6f}, {running_test_loss:.6f}], Early stop count = {early_stop_count}")
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = cfg.train.optimizer_params.lr * 0.5 ** ep
        """
    # save model
    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".pt")
    torch.save(model.state_dict(), filename)
    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".yaml")
    cfg.to_yaml(filename)
    logger.info("Finished training!")




if __name__ == "__main__":
    train()
