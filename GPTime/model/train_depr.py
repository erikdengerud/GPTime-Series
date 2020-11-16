import importlib
import sys
import logging
import torch
import os
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
sys.path.append("")

from GPTime.config import cfg
from GPTime.model.data import DeterministicTSDataSet

logger = logging.getLogger(__name__)

Criterion = getattr(importlib.import_module(cfg.train.criterion_module), cfg.train.criterion_name)
Optimizer = getattr(importlib.import_module(cfg.train.optimizer_module), cfg.train.optimizer_name)
Model = getattr(importlib.import_module(cfg.train.model_module), cfg.train.model_name)
Dataset = getattr(importlib.import_module(cfg.dataset.dataset_module), cfg.dataset.dataset_name)
DataLoader = getattr(importlib.import_module(cfg.train.dataloader_module), cfg.train.dataloader_name)

def train():
    if Model.__name__ == "MLP":
        model_params = cfg.train.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = cfg.train.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = cfg.train.model_params_tcn
    else:
        logger.warning("Unknown model name.")   
    model = Model(**model_params).double()
    logger.info(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    criterion = Criterion(**cfg.train.criterion_params)
    optimizer = Optimizer(model.parameters(), **cfg.train.optimizer_params)
    writer = SummaryWriter(log_dir=cfg.train.tensorboard_log_dir)
    
    # Dataset
    ds = Dataset(
        memory=model.memory,
        convolutions=True if Model.__name__=="TCN" else False,
        **cfg.dataset.dataset_params
        )
    train_length = int(ds.__len__() * cfg.train.train_set_size)
    val_length = int(ds.__len__() * cfg.train.val_set_size)
    test_length = ds.__len__() - train_length - val_length
    #train_ds, ds_val_tmp, ds_test_tmp  = random_split(
    """
    ds_train_tmp, ds_val_tmp, ds_test_tmp  = random_split(
        dataset=ds, 
        lengths=[train_length, val_length, test_length],
        generator=torch.torch.Generator()
        )
    train_ds = DeterministicTSDataSet(ds_train_tmp, num_tests_per_ts=cfg.train.num_tests_per_ts)
    val_ds = DeterministicTSDataSet(ds_val_tmp, num_tests_per_ts=cfg.train.num_tests_per_ts)
    test_ds = DeterministicTSDataSet(ds_test_tmp, num_tests_per_ts=cfg.train.num_tests_per_ts)
    """
    train_ds, val_ds, test_ds = random_split(
        dataset=ds, 
        lengths=[train_length, val_length, test_length],
        generator=torch.torch.Generator()
        )
    # Dataloader
    train_dl = DataLoader(train_ds, **cfg.train.dataloader_params)
    val_dl = DataLoader(val_ds, **cfg.train.dataloader_params)
    test_dl = DataLoader(train_ds, **cfg.train.dataloader_params)

    # training loop
    val_losses = []
    tenacity_count = 0
    for epoch in range(1, cfg.train.max_epochs+1):
        model.train()
        running_train_loss = 0.0
        for i, data in enumerate(train_dl):
            inputs, labels, freq = data
            optimizer.zero_grad()
            outputs = model(inputs)
            #logger.info(f"Shape of output: {outputs.shape}")
            labels = labels.unsqueeze(1)
            #logger.info(f"Shape of labels: {labels.shape}")
            assert outputs.shape == labels.shape, f"Output, {outputs.shape},  and labels, {labels.shape}, have different shapes."
            loss = criterion(outputs, labels)
            loss.backward()
            if cfg.train.clip_gradients:
                for p in model.parameters():
                    p.grad.data.clamp_(max=1e5, min=-1e5)
            optimizer.step()
            running_train_loss += loss.item()
        running_train_loss = running_train_loss / len(train_ds)
        writer.add_scalar("Loss/train", running_train_loss, epoch)
        

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                inputs, labels, freq = data
                outputs = model(inputs)
                labels = labels.unsqueeze(1)
                assert outputs.shape == labels.shape, f"Output, {outputs.shape},  and labels, {labels.shape}, have different shapes."
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        running_val_loss = running_val_loss / len(val_ds)
        writer.add_scalar("Loss/val", running_val_loss, epoch)
        
        
        running_test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                inputs, labels, freq = data
                outputs = model(inputs)
                labels = labels.unsqueeze(1)
                assert outputs.shape == labels.shape, f"Output, {outputs.shape},  and labels, {labels.shape}, have different shapes."
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
        running_test_loss = running_test_loss / len(test_ds)
        writer.add_scalar("Loss/test", running_test_loss , epoch)


        if epoch % 1 == 0: 
            logger.info(f"Epoch {epoch:>3d}: train_loss = {running_train_loss:.5f}, val_loss = {running_val_loss:.5f}, test_loss = {running_test_loss:.5f}")

        if epoch > cfg.train.early_stop_tenacity + 1:
            if running_val_loss < min(val_losses[-cfg.train.early_stop_tenacity :]):
                tenacity_count = 0
            else:
                tenacity_count += 1
        val_losses.append(running_val_loss)
        if tenacity_count >= cfg.train.early_stop_tenacity:
            break
        
    writer.close()
    # save model
    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".pt")
    torch.save(model.state_dict(), filename)
    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".yaml")
    cfg.to_yaml(filename)

    logger.info("Finished Training")

if __name__ == "__main__":
    train()