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

#from GPTime.model.data import TSDataset_iter
from GPTime.utils.scoring import score_M4, predict_M4
from GPTime.utils.metrics import MASE, SMAPE
from GPTime.utils.scaling import MASEScaler

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiplicativeLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR


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

def smape_2(forecast, target) -> np.ndarray:
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom

def train():

    num_epochs = 100
    learning_rate = 0.001
    scheduler_name = None#"multiplicative" # "plateau", "cosine", "cosine_warm"
    swa = True
    dropout = False 
    init_weights = False

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


    lr_decay_step = int(0.9*num_epochs) // 10
    if lr_decay_step == 0:
        lr_decay_step = 1


    #criterion = Criterion(**cfg.train.criterion_params)
    criterion = Criterion
    optimizer = Optimizer(model.parameters(), **cfg.train.optimizer_params)
    writer = SummaryWriter(log_dir=cfg.train.tensorboard_log_dir)

    if scheduler_name == "multiplicative":
        lmbda = lambda epoch: 0.95
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, "min")
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == "cosine_warm":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30)

    if swa:
        swa_model = AveragedModel(model)
        swa_start = int(0.9*num_epochs)
        swa_scheduler = SWALR(optimizer, swa_lr=0.000001)

    # Dataset
    ds = Dataset(
        memory=model.memory,
        convolutions=True if Model.__name__=="TCN" else False,
        **cfg.dataset.dataset_params
        )

    history_size_in_horizons = 1.5
    horizon = 18
    input_size = 4 * horizon
    maxscale = cfg.train.scale
    seasonal_init = cfg.train.seasonal_init
    print(f"seasonal init: {seasonal_init}")

    train_loader = DataLoader(dataset=ds, batch_size=1024)

    print("Training model.")
    print(f"Length of dataset: {len(ds)}")
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    running_loss = 0.0
    for ep in range(1, num_epochs+1):
        epoch_loss = 0.0
        batches_non_inf = 0
        for i, data in enumerate(train_loader):
            model.train()

            sample = data[0].to(device)
            label = data[1].to(device)
            sample_mask = data[2].to(device)
            label_mask = data[3].to(device)
            #last_period = data[4].to(device)
            freq_int = data[4].to(device)

            if seasonal_init:
                last_period = sample.shape[1]-freq_int
            else:
                last_period = torch.tensor(sample.shape[1]-1).repeat(sample.shape[0])

            optimizer.zero_grad()

            if maxscale:
                max_scale = torch.max(sample, 1).values.unsqueeze(1)                                                                                                                                    
                sample = torch.div(sample, max_scale)
                sample[torch.isnan(sample)] = 0.0

            forecast = model(sample, sample_mask, last_period)

            if maxscale:
                forecast = torch.mul(forecast, max_scale)
                sample = torch.mul(sample, max_scale)
            # TODO: Fix loss 
            training_loss = criterion(forecast, label, sample, sample_mask, freq_int)

            if np.isnan(float(training_loss)):
                print("Training loss is inf")
                print(i, data)
                break

            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batches_non_inf += 1

            epoch_loss += training_loss.item()

        running_loss += epoch_loss / batches_non_inf

        if swa and ep > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if scheduler_name == "plateau":
                scheduler.step(epoch_loss)
            elif scheduler_name is not None:
                scheduler.step()
            if scheduler_name is None:
                for param_group in optimizer.param_groups:
                    old_lr = param_group["lr"]
                    param_group["lr"] = learning_rate * 0.5 ** (ep // lr_decay_step)
                    new_lr = param_group["lr"]
                    if old_lr != new_lr:
                        print(f"Changed learning rate. Current lr = {param_group['lr']}")

        if (ep) % 5 == 0:
            print(f"Epoch {ep:<5d} [Avg. Loss, Loss] : [{running_loss / 5 :.4f}, {epoch_loss / batches_non_inf:.4f}]")
            running_loss = 0.0
    """
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
    """
    # Build forecasts

    """
    df_train = pd.read_csv(os.path.join(cfg.source.path.M4.raw_train, "Monthly-train.csv"), index_col=0)
    all_ts = [v[~np.isnan(v)] for v in df_train.values]
    training_set = TSDataset_iter(all_ts)
    train_data, mask = training_set.last_insample_window()

    model.eval()
    with torch.no_grad():
        for i in range(18):
            sample = torch.from_numpy(train_data[:, -model.memory:])
            sample_mask = torch.from_numpy(mask[:,-model.memory:])
            out = model(sample, sample_mask, 12).cpu().detach().numpy()
            train_data = np.hstack((train_data, out))
            mask = np.hstack((mask, np.ones((mask.shape[0], 1))))
        forecast = train_data[:, -18:]

    df_test_monthly = pd.read_csv(os.path.join(cfg.source.path.M4.raw_test, "Monthly-test.csv"), index_col=0)
    df_train_monthly = pd.read_csv(os.path.join(cfg.source.path.M4.raw_train, "Monthly-train.csv"), index_col=0)

    #logger.debug(f"np.sum(training_set.last_insample_window() - df_train_monthly.values) = {np.sum(training_set.last_insample_window() - df_train_monthly.values)}")
    in_sample = np.array([ts[~pd.isnull(ts)] for ts in df_train_monthly.values], dtype=object)
    target = np.array([ts[~pd.isnull(ts)] for ts in df_test_monthly.values], dtype=object)
    assert forecast.shape == target.shape, f"forecast.shape: {forecast.shape}, target.shape: {target.shape}"
    mase_res = np.mean([mase(forecast=forecast[i], insample=in_sample[i], outsample=target[i], frequency=12) for i in range(len(forecast))])
    smape_res = np.mean([smape_2(forecast=forecast[i], target=target[i]) for i in range(len(forecast))])
    logger.debug(f"MASE nbeats way = {mase_res}")
    logger.debug(f"SMAPE nbeats way = {smape_res}")


    scale = MASEScaler().fit(df_train_monthly.values, freq=12).scale_.flatten()
    mase_me = MASE(target, forecast, scale)
    smape_me = SMAPE(target, forecast)

    logger.debug(f"MASE me = {mase_me}")
    logger.debug(f"SMAPE me = {smape_me}")
    """
    """
    preds = predict_M4(model=model, scale=maxscale)
    #logger.debug(f"np.sum(training_set.last_insample_window() - X) = {np.sum(training_set.last_insample_window()[0] - X)}")
    #logger.debug(f"np.sum(preds - forecast) = {np.sum(preds - forecast)}")
    res = score_M4(predictions=preds)
    logger.debug(res)
    """

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

    if swa:
        filename = os.path.join(cfg.train.model_save_path, cfg.run.name + "_swa.pt")
        torch.save(model.state_dict(), filename)
        preds = predict_M4(model=swa_model, scale=maxscale, seasonal_init=seasonal_init)
        res = score_M4(predictions=preds)
        logger.info(res)
        logger.info("Finished SWA!")




if __name__ == "__main__":
    train()
