import torch
import torch.nn as nn

import importlib

import logging
import os
import numpy as np

from GPTime.utils.scoring import score_M4, predict_M4
from GPTime.utils.metrics import MASE, SMAPE
from GPTime.utils.scaling import MASEScaler

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts # MultiplicativeLR

logger = logging.getLogger(__name__)

def finetune(finetune_cfg):
    logger.debug("FINETUNE!")
    
    Model = getattr(importlib.import_module(finetune_cfg.model_module), finetune_cfg.model_name)
    Criterion = getattr(
        importlib.import_module(finetune_cfg.criterion_module), finetune_cfg.criterion_name
    )
    Optimizer = getattr(
        importlib.import_module(finetune_cfg.optimizer_module), finetune_cfg.optimizer_name
    )
    Dataset = getattr(
        importlib.import_module(finetune_cfg.dataset_module), finetune_cfg.dataset_name
    )
    DataLoader = getattr(
        importlib.import_module(finetune_cfg.dataloader_module), finetune_cfg.dataloader_name
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    # load model
    
    model = Model(**finetune_cfg.model_params_mlp)
    model.load_state_dict(torch.load(finetune_cfg.model_path))
    logger.info(
        f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    # Set all parameters to require grad false
    for param in model.parameters():
        param.requires_grad = False
    logger.info(
        f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    model.double()
    model.to(device)
    preds, df_preds = predict_M4(model=model, scale=finetune_cfg.scale, seasonal_init=finetune_cfg.seasonal_init, val_set=finetune_cfg.val, encode_frequencies=finetune_cfg.model_params_mlp.encode_frequencies)
    res = score_M4(predictions=preds, val=finetune_cfg.val)
    logger.info(res)
    preds, df_preds = predict_M4(model=model, scale=finetune_cfg.scale, seasonal_init=finetune_cfg.seasonal_init, encode_frequencies=finetune_cfg.model_params_mlp.encode_frequencies)
    res = score_M4(predictions=preds)
    logger.info(res)
    # set the out layer to a new layer, or last N layers. use require grad=True
    model.layers[-1] = nn.Linear(in_features=1024, out_features=1024)
    model.out = nn.Linear(in_features=1024, out_features=1)
    model.double()
    model.to(device)
    logger.info(
        f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    criterion = Criterion
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer = Optimizer(params_to_update, **finetune_cfg.optimizer_params)

    # Learning rate 
    num_lr_steps = 10
    logger.info(f"{num_lr_steps} steps in the learning schedule if None")
    lr_decay_step = int(finetune_cfg.max_epochs_1) // num_lr_steps
    if lr_decay_step == 0:
        lr_decay_step = 1

    if finetune_cfg.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, "min", verbose=True, patience=finetune_cfg.patience)
    logger.info(f"Using learning rate sheduler: {finetune_cfg.lr_scheduler}")

    # load dataset, with validation set

    if finetune_cfg.val:
        logger.info("Using a validation split.")
        # Make a train and val set
        ds_train = Dataset(
            memory=model.memory,
            convolutions=True if Model.__name__=="TCN" else False,
            ds_type="train",
            **finetune_cfg.dataset_params
            )

        ds_val = Dataset(
            memory=model.memory,
            convolutions=True if Model.__name__=="TCN" else False,
            ds_type="val",
            **finetune_cfg.dataset_params
            )

    else:
        logger.info("Not using a validation set. Training on the full dataset.")
        ds_train = Dataset(
            memory=model.memory,
            convolutions=True if Model.__name__=="TCN" else False,
            ds_type="full",
            **finetune_cfg.dataset_params
        )

    train_loader = DataLoader(ds_train, **finetune_cfg.dataloader_params)
    if finetune_cfg.val:
        val_loader = DataLoader(ds_train, **finetune_cfg.dataloader_params)

    # Train the final layer(s) until val stops
    # Use lower learning rate

    running_loss = 0.0
    val_running_loss = 0.0
    low_loss = np.inf
    early_stop_count = 0
    for ep in range(1, finetune_cfg.max_epochs_1+1):
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
            if finetune_cfg.seasonal_init:
                last_period = sample.shape[1]-freq_int
            else:
                last_period = torch.tensor(sample.shape[1]-1).repeat(sample.shape[0]).to(device)

            optimizer.zero_grad()

            if finetune_cfg.scale:
                max_scale = torch.max(sample, 1).values.unsqueeze(1)
                if len((max_scale == 0).nonzero()) > 0:
                    zero_idx = (max_scale==0).nonzero()
                    max_scale[zero_idx[:,0], zero_idx[:,1]] = 1.0
                sample = torch.div(sample, max_scale)
                sample[torch.isnan(sample)] = 0.0

            forecast = model(sample, sample_mask, last_period, freq_str_arr)

            if finetune_cfg.scale:
                forecast = torch.mul(forecast, max_scale)
                sample = torch.mul(sample, max_scale)

            training_loss = criterion(forecast, label, sample, sample_mask, freq_int)

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

        if finetune_cfg.val:
            val_epoch_loss = 0.0
            val_batches_non_inf = 0
            for i, data in enumerate(val_loader):
                model.eval()
                sample = data[0].to(device)
                label = data[1].to(device)
                sample_mask = data[2].to(device)
                label_mask = data[3].to(device)
                freq_int = data[4].to(device)
                freq_str_arr = np.expand_dims(np.array(data[5]), axis=1)

                if finetune_cfg.seasonal_init:
                    last_period = sample.shape[1]-freq_int
                else:
                    last_period = torch.tensor(sample.shape[1]-1).repeat(sample.shape[0]).to(device)

                if finetune_cfg.scale:
                    max_scale = torch.max(sample, 1).values.unsqueeze(1)
                    sample = torch.div(sample, max_scale)
                    sample[torch.isnan(sample)] = 0.0

                forecast = model(sample, sample_mask, last_period, freq_str_arr)

                if finetune_cfg.scale:
                    forecast = torch.mul(forecast, max_scale)
                    sample = torch.mul(sample, max_scale)

                val_loss = criterion(forecast, label, sample, sample_mask, freq_int)

                val_batches_non_inf += 1
                val_epoch_loss += val_loss.item()

            val_running_loss += val_epoch_loss / val_batches_non_inf

        if finetune_cfg.val:
            if val_epoch_loss < low_loss:
                low_loss = val_epoch_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count > finetune_cfg.early_stop_tenacity:
                logger.info(f"Early stop after epoch {ep}.")
                break

        if finetune_cfg.lr_scheduler == "plateau":
            if finetune_cfg.val:
                scheduler.step(val_epoch_loss)
            else:
                scheduler.step(epoch_loss)
        elif finetune_cfg.lr_scheduler is not None:
            scheduler.step()
        if finetune_cfg.lr_scheduler is None:
            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                param_group["lr"] = finetune_cfg.optimizer_params.lr * 0.5 ** (ep // lr_decay_step)
                new_lr = param_group["lr"]
                if old_lr != new_lr:
                    logger.info(f"Changed learning rate. Current lr = {param_group['lr']}")
        
        if (ep) % finetune_cfg.log_freq == 0:
            if finetune_cfg.val:
                logger.info(f"Epoch {ep:<5d} [Avg. Loss, Loss], [avg. ValLoss, ValLoss]: [{running_loss / finetune_cfg.log_freq :.4f}, {epoch_loss / batches_non_inf:.4f}] [{val_running_loss / finetune_cfg.log_freq :.4f}, {val_epoch_loss / val_batches_non_inf:.4f}], {early_stop_count}")
                running_loss = 0.0
                val_running_loss = 0.0
            else:
                logger.info(f"Epoch {ep:<5d} [Avg. Loss, Loss]: [{running_loss / finetune_cfg.log_freq :.4f}, {epoch_loss / batches_non_inf:.4f}], {early_stop_count}")
                running_loss = 0.0
    
    forecast_path = os.path.join(finetune_cfg.model_save_path, f"{finetune_cfg.name}_step1_forecast.csv")
    result_path = os.path.join(finetune_cfg.model_save_path, f"{finetune_cfg.name}_step1_result.csv")
    preds, df_preds = predict_M4(model=model, scale=finetune_cfg.scale, seasonal_init=finetune_cfg.seasonal_init, val_set=finetune_cfg.val, encode_frequencies=finetune_cfg.model_params_mlp.encode_frequencies)
    res = score_M4(predictions=preds, df_results_name=result_path, val=finetune_cfg.val)
    logger.info(res)

    df_preds.to_csv(forecast_path)


    # set the out layer to a new layer, or last N layers. use require grad=True
    logger.info(
        f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    for param in model.parameters():
        param.requires_grad = True
    logger.info(
        f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    optimizer = Optimizer(model.parameters(), **finetune_cfg.optimizer_params)
    criterion = Criterion

    # Learning rate 
    num_lr_steps = 10
    logger.info(f"{num_lr_steps} steps in the learning schedule if None")
    lr_decay_step = int(finetune_cfg.max_epochs_2) // num_lr_steps
    if lr_decay_step == 0:
        lr_decay_step = 1

    if finetune_cfg.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, "min", verbose=True, patience=finetune_cfg.patience)
    logger.info(f"Using learning rate sheduler: {finetune_cfg.lr_scheduler}")

    running_loss = 0.0
    val_running_loss = 0.0
    low_loss = np.inf
    early_stop_count = 0
    for ep in range(1, finetune_cfg.max_epochs_2+1):
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
            if finetune_cfg.seasonal_init:
                last_period = sample.shape[1]-freq_int
            else:
                last_period = torch.tensor(sample.shape[1]-1).repeat(sample.shape[0]).to(device)

            optimizer.zero_grad()

            if finetune_cfg.scale:
                max_scale = torch.max(sample, 1).values.unsqueeze(1)
                if len((max_scale == 0).nonzero()) > 0:
                    zero_idx = (max_scale==0).nonzero()
                    max_scale[zero_idx[:,0], zero_idx[:,1]] = 1.0
                sample = torch.div(sample, max_scale)
                sample[torch.isnan(sample)] = 0.0

            forecast = model(sample, sample_mask, last_period, freq_str_arr)

            if finetune_cfg.scale:
                forecast = torch.mul(forecast, max_scale)
                sample = torch.mul(sample, max_scale)

            training_loss = criterion(forecast, label, sample, sample_mask, freq_int)

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

        if finetune_cfg.val:
            val_epoch_loss = 0.0
            val_batches_non_inf = 0
            for i, data in enumerate(val_loader):
                model.eval()
                sample = data[0].to(device)
                label = data[1].to(device)
                sample_mask = data[2].to(device)
                label_mask = data[3].to(device)
                freq_int = data[4].to(device)
                freq_str_arr = np.expand_dims(np.array(data[5]), axis=1)

                if finetune_cfg.seasonal_init:
                    last_period = sample.shape[1]-freq_int
                else:
                    last_period = torch.tensor(sample.shape[1]-1).repeat(sample.shape[0]).to(device)

                if finetune_cfg.scale:
                    max_scale = torch.max(sample, 1).values.unsqueeze(1)
                    sample = torch.div(sample, max_scale)
                    sample[torch.isnan(sample)] = 0.0

                forecast = model(sample, sample_mask, last_period, freq_str_arr)

                if finetune_cfg.scale:
                    forecast = torch.mul(forecast, max_scale)
                    sample = torch.mul(sample, max_scale)

                val_loss = criterion(forecast, label, sample, sample_mask, freq_int)

                val_batches_non_inf += 1
                val_epoch_loss += val_loss.item()

            val_running_loss += val_epoch_loss / val_batches_non_inf

        if finetune_cfg.val:
            if val_epoch_loss < low_loss:
                low_loss = val_epoch_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count > finetune_cfg.early_stop_tenacity:
                logger.info(f"Early stop after epoch {ep}.")
                break

        if finetune_cfg.lr_scheduler == "plateau":
            if finetune_cfg.val:
                scheduler.step(val_epoch_loss)
            else:
                scheduler.step(epoch_loss)
        elif finetune_cfg.lr_scheduler is not None:
            scheduler.step()
        if finetune_cfg.lr_scheduler is None:
            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                param_group["lr"] = finetune_cfg.optimizer_params.lr * 0.5 ** (ep // lr_decay_step)
                new_lr = param_group["lr"]
                if old_lr != new_lr:
                    logger.info(f"Changed learning rate. Current lr = {param_group['lr']}")
        
        if (ep) % finetune_cfg.log_freq == 0:
            if finetune_cfg.val:
                logger.info(f"Epoch {ep:<5d} [Avg. Loss, Loss], [avg. ValLoss, ValLoss]: [{running_loss / finetune_cfg.log_freq :.4f}, {epoch_loss / batches_non_inf:.4f}] [{val_running_loss / finetune_cfg.log_freq :.4f}, {val_epoch_loss / val_batches_non_inf:.4f}], {early_stop_count}")
                running_loss = 0.0
                val_running_loss = 0.0
            else:
                logger.info(f"Epoch {ep:<5d} [Avg. Loss, Loss]: [{running_loss / finetune_cfg.log_freq :.4f}, {epoch_loss / batches_non_inf:.4f}], {early_stop_count}")
                running_loss = 0.0
    
    forecast_path = os.path.join(finetune_cfg.model_save_path, f"{finetune_cfg.name}_step2_forecast.csv")
    result_path = os.path.join(finetune_cfg.model_save_path, f"{finetune_cfg.name}_step2_result.csv")
    preds, df_preds = predict_M4(model=model, scale=finetune_cfg.scale, seasonal_init=finetune_cfg.seasonal_init, val_set=finetune_cfg.val, encode_frequencies=finetune_cfg.model_params_mlp.encode_frequencies)
    res = score_M4(predictions=preds, df_results_name=result_path, val=finetune_cfg.val)
    logger.info(res)

    df_preds.to_csv(forecast_path)


    model = Model(**finetune_cfg.model_params_mlp)
    model.load_state_dict(torch.load(finetune_cfg.model_path))
    model.double()
    model.to(device)
    logger.info(
        f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    criterion = Criterion
    optimizer = Optimizer(model.parameters(), **finetune_cfg.optimizer_params)

    # Learning rate 
    num_lr_steps = 10
    logger.info(f"{num_lr_steps} steps in the learning schedule if None")
    lr_decay_step = int(finetune_cfg.max_epochs_3) // num_lr_steps
    if lr_decay_step == 0:
        lr_decay_step = 1

    if finetune_cfg.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, "min", verbose=True, patience=finetune_cfg.patience)
    logger.info(f"Using learning rate sheduler: {finetune_cfg.lr_scheduler}")

    running_loss = 0.0
    val_running_loss = 0.0
    low_loss = np.inf
    early_stop_count = 0
    for ep in range(1, finetune_cfg.max_epochs_3+1):
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
            if finetune_cfg.seasonal_init:
                last_period = sample.shape[1]-freq_int
            else:
                last_period = torch.tensor(sample.shape[1]-1).repeat(sample.shape[0]).to(device)

            optimizer.zero_grad()

            if finetune_cfg.scale:
                max_scale = torch.max(sample, 1).values.unsqueeze(1)
                if len((max_scale == 0).nonzero()) > 0:
                    zero_idx = (max_scale==0).nonzero()
                    max_scale[zero_idx[:,0], zero_idx[:,1]] = 1.0
                sample = torch.div(sample, max_scale)
                sample[torch.isnan(sample)] = 0.0

            forecast = model(sample, sample_mask, last_period, freq_str_arr)

            if finetune_cfg.scale:
                forecast = torch.mul(forecast, max_scale)
                sample = torch.mul(sample, max_scale)

            training_loss = criterion(forecast, label, sample, sample_mask, freq_int)

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

        if finetune_cfg.val:
            val_epoch_loss = 0.0
            val_batches_non_inf = 0
            for i, data in enumerate(val_loader):
                model.eval()
                sample = data[0].to(device)
                label = data[1].to(device)
                sample_mask = data[2].to(device)
                label_mask = data[3].to(device)
                freq_int = data[4].to(device)
                freq_str_arr = np.expand_dims(np.array(data[5]), axis=1)

                if finetune_cfg.seasonal_init:
                    last_period = sample.shape[1]-freq_int
                else:
                    last_period = torch.tensor(sample.shape[1]-1).repeat(sample.shape[0]).to(device)

                if finetune_cfg.scale:
                    max_scale = torch.max(sample, 1).values.unsqueeze(1)
                    sample = torch.div(sample, max_scale)
                    sample[torch.isnan(sample)] = 0.0

                forecast = model(sample, sample_mask, last_period, freq_str_arr)

                if finetune_cfg.scale:
                    forecast = torch.mul(forecast, max_scale)
                    sample = torch.mul(sample, max_scale)

                val_loss = criterion(forecast, label, sample, sample_mask, freq_int)

                val_batches_non_inf += 1
                val_epoch_loss += val_loss.item()

            val_running_loss += val_epoch_loss / val_batches_non_inf

        if finetune_cfg.val:
            if val_epoch_loss < low_loss:
                low_loss = val_epoch_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count > finetune_cfg.early_stop_tenacity:
                logger.info(f"Early stop after epoch {ep}.")
                break

        if finetune_cfg.lr_scheduler == "plateau":
            if finetune_cfg.val:
                scheduler.step(val_epoch_loss)
            else:
                scheduler.step(epoch_loss)
        elif finetune_cfg.lr_scheduler is not None:
            scheduler.step()
        if finetune_cfg.lr_scheduler is None:
            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                param_group["lr"] = finetune_cfg.optimizer_params.lr * 0.5 ** (ep // lr_decay_step)
                new_lr = param_group["lr"]
                if old_lr != new_lr:
                    logger.info(f"Changed learning rate. Current lr = {param_group['lr']}")
        
        if (ep) % finetune_cfg.log_freq == 0:
            if finetune_cfg.val:
                logger.info(f"Epoch {ep:<5d} [Avg. Loss, Loss], [avg. ValLoss, ValLoss]: [{running_loss / finetune_cfg.log_freq :.4f}, {epoch_loss / batches_non_inf:.4f}] [{val_running_loss / finetune_cfg.log_freq :.4f}, {val_epoch_loss / val_batches_non_inf:.4f}], {early_stop_count}")
                running_loss = 0.0
                val_running_loss = 0.0
            else:
                logger.info(f"Epoch {ep:<5d} [Avg. Loss, Loss]: [{running_loss / finetune_cfg.log_freq :.4f}, {epoch_loss / batches_non_inf:.4f}], {early_stop_count}")
                running_loss = 0.0
    
    forecast_path = os.path.join(finetune_cfg.model_save_path, f"{finetune_cfg.name}_initialization_forecast.csv")
    result_path = os.path.join(finetune_cfg.model_save_path, f"{finetune_cfg.name}_initialization_result.csv")
    preds, df_preds = predict_M4(model=model, scale=finetune_cfg.scale, seasonal_init=finetune_cfg.seasonal_init, val_set=finetune_cfg.val, encode_frequencies=finetune_cfg.model_params_mlp.encode_frequencies)
    res = score_M4(predictions=preds, df_results_name=result_path, val=finetune_cfg.val)
    logger.info(res)
    df_preds.to_csv(forecast_path)

    logger.info("Finished training!")
