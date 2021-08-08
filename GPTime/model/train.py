import importlib
import logging
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import os
import copy
import numpy as np
import pandas as pd
import time
import sys

sys.path.append("")

from GPTime.config import cfg

from GPTime.utils.scoring import score_M4, predict_M4
from GPTime.utils.metrics import MASE, SMAPE
from GPTime.utils.scaling import MASEScaler

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts # MultiplicativeLR
#from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Subset, ConcatDataset


logger = logging.getLogger(__name__)


#def smape_2_loss(forecast, target, mask) -> t.float:
#    """
#    sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)
#    :param forecast: Forecast values. Shape: batch, time
#    :param target: Target values. Shape: batch, time
#    :param mask: 0/1 mask. Shape: batch, time
#    :return: Loss value
#    """
#    return 200 * t.mean(divide_no_nan(t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)) * mask)


def train(train_cfg):
    #np.random.seed(1729)
    #torch.manual_seed(1729)
    Model = getattr(importlib.import_module(train_cfg.model_module), train_cfg.model_name)
    Criterion = getattr(
        importlib.import_module(train_cfg.criterion_module), train_cfg.criterion_name
    )
    Optimizer = getattr(
        importlib.import_module(train_cfg.optimizer_module), train_cfg.optimizer_name
    )
    Dataset = getattr(
        importlib.import_module(train_cfg.dataset_module), train_cfg.dataset_name
    )
    DataLoader = getattr(
        importlib.import_module(train_cfg.dataloader_module), train_cfg.dataloader_name
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if Model.__name__ == "MLP":
        model_params = train_cfg.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = train_cfg.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = train_cfg.model_params_tcn
    else:
        logger.warning("Unknown model name.")
    model = Model(**model_params).double()
    model.to(device)
    logger.info(
        f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    #criterion = Criterion(**train_cfg.criterion_params)
    criterion = Criterion
    optimizer = Optimizer(model.parameters(), **train_cfg.optimizer_params)
    writer = SummaryWriter(log_dir=train_cfg.tensorboard_log_dir)

    # Learning rate 
    num_lr_steps = 10
    logger.info(f"{num_lr_steps} steps in the learning schedule if None")
    lr_decay_step = int(train_cfg.max_epochs) // num_lr_steps
    if lr_decay_step == 0:
        lr_decay_step = 1

    if train_cfg.lr_scheduler == "multiplicative":
        lmbda = lambda epoch: 0.95
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda, verbose=True)
    elif train_cfg.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, "min", verbose=True, patience=train_cfg.patience)
    elif train_cfg.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.max_epochs, eta_min=0.00000001)
    elif train_cfg.lr_scheduler == "cosine_warm":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, verbose=True)
    logger.info(f"Using learning rate sheduler: {train_cfg.lr_scheduler}")

    if train_cfg.swa:
        # TODO: Fix swa to work with early stop. Not really a problem without  the validatoin set. 
        swa_model = AveragedModel(model)
        swa_start = int(0.9*train_cfg.max_epochs)
        swa_scheduler = SWALR(optimizer, swa_lr=0.000001)

    # Dataset
    # TODO: log lookback and loss function
    if not train_cfg.more_data:
        if train_cfg.val:
            logger.info("Using a validation split.")
            assert train_cfg.h_val + train_cfg.v_val < 2, "Horizontal and vertical validation split both selected!"
            # Make a train and val set
            if train_cfg.v_val:
                # make a horisontal train val split
                ds_train = Dataset(
                    memory=model.memory,
                    convolutions=True if Model.__name__=="TCN" else False,
                    ds_type="train",
                    **train_cfg.dataset_params
                    )

                ds_val = Dataset(
                    memory=model.memory,
                    convolutions=True if Model.__name__=="TCN" else False,
                    ds_type="val",
                    **train_cfg.dataset_params
                    )

            elif train_cfg.h_val:
                # make a vertical train val split
                # Proportion of ds to use
                assert train_cfg.proportion <= 1 and train_cfg.proportion > 0, "Proportion of dataset not between 0 and 1."
                proportion_length = int(ds.__len__() * train_cfg.proportion)
                ds_use, _ = random_split(dataset=ds, lengths=[proportion_length, ds.__len__() - proportion_length])

                train_length = int(ds_use.__len__() * train_cfg.train_set_size)
                val_length = ds_use.__len__() - train_length

                train_ds, val_ds = random_split(
                    dataset=ds_use, 
                    lengths=[train_length, val_length],
                    #generator=torch.torch.Generator()
                )
                logger.info(f"Using {train_cfg.proportion * 100}% of the available dataset.")
                logger.info(f"Using frequencies: {[freq for freq, true_false in train_cfg.dataset_params.frequencies.items() if true_false]}")
                logger.info(f"Train size: {train_ds.__len__()}, Val size: {val_ds.__len__()}")

                # Dataloader
                train_loader = DataLoader(train_ds, **train_cfg.dataloader_params)
                val_loader = DataLoader(val_ds, **train_cfg.dataloader_params)
                test_loader = DataLoader(train_ds, **train_cfg.dataloader_params)

            else:
                # Not specified
                logger.warning("Type of train val split not specified!")
                raise Warning
        else:
            logger.info("Not using a validation set. Training on the full dataset.")
            ds_train = Dataset(
                memory=model.memory,
                convolutions=True if Model.__name__=="TCN" else False,
                ds_type="full",
                **train_cfg.dataset_params
            )

        logger.info(f"seasonal init: {train_cfg.seasonal_init}")

        train_loader = DataLoader(dataset=ds_train, **train_cfg.dataloader_params)
        if train_cfg.val:
            val_loader = DataLoader(dataset=ds_val, **train_cfg.dataloader_params)
        logger.info("Training model.")
        logger.info(f"Length of dataset: {len(ds_train)}")

    else:
        # Make one dataset for each frequency
        logger.debug(f"proportion used: {train_cfg.proportion}")
        datasets = []
        dataset_paths = {}
        m4_path_dict = {}
        for ds_name in train_cfg.dataset_params.dataset_paths.keys():
            if ds_name != "M4":
                dataset_paths[ds_name] = train_cfg.dataset_params.dataset_paths[ds_name]
            else:
                m4_path_dict[ds_name] = train_cfg.dataset_params.dataset_paths[ds_name]
        logger.debug(f"dataset_paths: {dataset_paths}")
        for freq in train_cfg.dataset_params.frequencies.keys():
            if train_cfg.dataset_params.frequencies[freq]:
                #logger.debug(freq)
                use_frequencies = {}
                for freq_set in train_cfg.dataset_params.frequencies.keys():
                    if freq_set == freq:
                        use_frequencies[freq_set] = True
                    else:
                        use_frequencies[freq_set] = False
                tmp_params = copy.copy(train_cfg.dataset_params)
                tmp_params.frequencies = use_frequencies
                tmp_params.dataset_paths = dataset_paths
                ds_freq = Dataset(
                    memory=model.memory,
                    convolutions=False,
                    **tmp_params,
                    )
                datasets.append(ds_freq)
        prop_datasets = []
        for ds in datasets:
            split = int(len(ds)*train_cfg.proportion)
            #logger.debug(f"len(ds): {len(ds)}")
            #logger.debug(f"split idx: {split}")
            indices = list(range(len(ds)))
            np.random.seed(1729)
            np.random.shuffle(indices)
            keep_indices = indices[:split]
            #logger.debug(f"len(keep_indices): {len(keep_indices)}")
            prop_ds = Subset(dataset=ds, indices=keep_indices)
            #logger.debug(f"len(prop_ds): {len(prop_ds)}")
            prop_datasets.append(prop_ds)
        if len(m4_path_dict):
            tmp_params = copy.copy(train_cfg.dataset_params)
            tmp_params.dataset_paths = m4_path_dict
            m4_ds = Dataset(
                memory=model.memory,
                convolutions=False,
                **tmp_params,
                )
            prop_datasets.append(m4_ds)
        concat_ds = ConcatDataset(prop_datasets)
 
        logger.debug(f"len(concat_ds): {len(concat_ds)}")
        train_loader = DataLoader(concat_ds, **train_cfg.dataloader_params)

    running_loss = 0.0
    val_running_loss = 0.0
    low_loss = np.inf
    early_stop_count = 0
    for ep in range(1, train_cfg.max_epochs+1):
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
            freq_str_arr = np.expand_dims(np.array(data[5]), axis=1)

            if train_cfg.seasonal_init:
                last_period = sample.shape[1]-freq_int
            else:
                last_period = torch.tensor(sample.shape[1]-1).repeat(sample.shape[0]).to(device)

            optimizer.zero_grad()

            if train_cfg.scale:
                max_scale = torch.max(sample, 1).values.unsqueeze(1)
                if len((max_scale == 0).nonzero()) > 0:
                    zero_idx = (max_scale==0).nonzero()
                    max_scale[zero_idx[:,0], zero_idx[:,1]] = 1.0
                sample = torch.div(sample, max_scale)
                sample[torch.isnan(sample)] = 0.0

            forecast = model(sample, sample_mask, last_period, freq_str_arr)

            if train_cfg.scale:
                forecast = torch.mul(forecast, max_scale)
                sample = torch.mul(sample, max_scale)

            training_loss = criterion(forecast, label, sample, sample_mask, freq_int)
            #training_loss = smape_2_loss(forecast, label, sample, sample_mask, freq_int)

            if np.isnan(float(training_loss)):
                logger.warning("Training loss is inf")
                logger.debug(i, data)
                break

            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batches_non_inf += 1

            epoch_loss += training_loss.item()

        running_loss += epoch_loss / batches_non_inf

        if train_cfg.val:
            val_epoch_loss = 0.0
            val_batches_non_inf = 0
            for i, data in enumerate(val_loader):
                model.eval()
                sample = data[0].to(device)
                label = data[1].to(device)
                sample_mask = data[2].to(device)
                label_mask = data[3].to(device)
                #last_period = data[4].to(device)
                freq_int = data[4].to(device)
                freq_str_arr = np.expand_dims(np.array(data[5]), axis=1)

                if train_cfg.seasonal_init:
                    last_period = sample.shape[1]-freq_int
                else:
                    last_period = torch.tensor(sample.shape[1]-1).repeat(sample.shape[0]).to(device)

                if train_cfg.scale:
                    max_scale = torch.max(sample, 1).values.unsqueeze(1)
                    sample = torch.div(sample, max_scale)
                    sample[torch.isnan(sample)] = 0.0

                forecast = model(sample, sample_mask, last_period, freq_str_arr)
                #forecast = model(sample, sample_mask, last_period)

                if train_cfg.scale:
                    forecast = torch.mul(forecast, max_scale)
                    sample = torch.mul(sample, max_scale)
                # TODO: Fix loss 
                val_loss = criterion(forecast, label, sample, sample_mask, freq_int)

                val_batches_non_inf += 1
                val_epoch_loss += val_loss.item()

            val_running_loss += val_epoch_loss / val_batches_non_inf

        if train_cfg.val:
            if val_epoch_loss < low_loss:
                low_loss = val_epoch_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count > train_cfg.early_stop_tenacity:
                logger.info(f"Early stop after epoch {ep}.")
                break

        if train_cfg.swa and ep > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if train_cfg.lr_scheduler == "plateau":
                if train_cfg.val:
                    scheduler.step(val_epoch_loss)
                else:
                    scheduler.step(epoch_loss)
            elif train_cfg.lr_scheduler is not None:
                scheduler.step()
            if train_cfg.lr_scheduler is None:
                for param_group in optimizer.param_groups:
                    old_lr = param_group["lr"]
                    param_group["lr"] = train_cfg.optimizer_params.lr * 0.5 ** (ep // lr_decay_step)
                    new_lr = param_group["lr"]
                    if old_lr != new_lr:
                        logger.info(f"Changed learning rate. Current lr = {param_group['lr']}")
        
        if (ep) % train_cfg.log_freq == 0:
            if train_cfg.val:
                logger.info(f"Epoch {ep:<5d} [Avg. Loss, Loss], [avg. ValLoss, ValLoss]: [{running_loss / train_cfg.log_freq :.4f}, {epoch_loss / batches_non_inf:.4f}] [{val_running_loss / train_cfg.log_freq :.4f}, {val_epoch_loss / val_batches_non_inf:.4f}], {early_stop_count}")
                running_loss = 0.0
                val_running_loss = 0.0
            else:
                logger.info(f"Epoch {ep:<5d} [Avg. Loss, Loss]: [{running_loss / train_cfg.log_freq :.4f}, {epoch_loss / batches_non_inf:.4f}], {early_stop_count}")
                running_loss = 0.0
    
    preds, df_preds = predict_M4(model=model, scale=train_cfg.scale, seasonal_init=train_cfg.seasonal_init, encode_frequencies=train_cfg.model_params_mlp.encode_frequencies)
    res = score_M4(predictions=preds)
    logger.info(res)
    # save model
    filename = os.path.join(train_cfg.model_save_path, train_cfg.name + ".pt")
    os.makedirs(os.path.join(train_cfg.model_save_path, train_cfg.name))
    torch.save(model.state_dict(), filename)
    #filename = os.path.join(train_cfg.model_save_path, train_cfg.name + ".yml")
    #train_cfg.to_yaml(filename)
    # vv comment out
    #preds, df_preds = predict_M4(model=model, scale=train_cfg.scale, seasonal_init=train_cfg.seasonal_init)
    #res = score_M4(predictions=preds)
    #logger.info(res)
    logger.info("Finished training!")

    if train_cfg.swa:
        filename = os.path.join(train_cfg.model_save_path, train_cfg.name + "_swa.pt")
        torch.save(model.state_dict(), filename)
        preds = predict_M4(model=swa_model, scale=train_cfg.scale, seasonal_init=train_cfg.seasonal_init)
        res = score_M4(predictions=preds)
        logger.info(res)
        logger.info("Finished SWA!")


if __name__ == "__main__":
    train()
