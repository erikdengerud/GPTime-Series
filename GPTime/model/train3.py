import importlib
import logging
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import pandas as pd
import sys

sys.path.append("")




from GPTime.config import cfg
from GPTime.model.data import DeterministicTSDataSet



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


def train3():
    #np.random.seed(1729)
    #torch.manual_seed(1729)
    if Model.__name__ == "MLP":
        model_params = cfg.train.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = cfg.train.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = cfg.train.model_params_tcn
    else:
        logger.warning("Unknown model name.")
    model = Model(**model_params).double()
    logger.info(
        f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
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

    train_ds, val_ds, test_ds = random_split(
        dataset=ds, 
        lengths=[train_length, val_length, test_length],
        generator=torch.torch.Generator()
    )

    # Dataloader
    trainloader = DataLoader(train_ds, **cfg.train.dataloader_params)
    valloader = DataLoader(val_ds, **cfg.train.dataloader_params)
    testloader = DataLoader(train_ds, **cfg.train.dataloader_params)

    early_stop_count = 0
    low_loss = np.inf
    for ep in range(1, cfg.train.max_epochs + 1):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0], data[1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #running_loss /= len(train_ds)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                inputs, labels = data[0], data[1]
                outputs = model(inputs)
                loss = criterion(outputs.flatten(), labels)
                running_val_loss += loss.item()
            if running_val_loss < low_loss:
                early_stop_count = 0
                low_loss = running_val_loss
            else:
                early_stop_count += 1
            if early_stop_count > cfg.train.early_stop_tenacity:
                print(f"Early stop after epoch {ep}.")
                break
        #running_val_loss /= len(val_ds)

        running_test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels, freq = data
                outputs = model(inputs)
                labels = labels.unsqueeze(1)
                assert outputs.shape == labels.shape, f"Output, {outputs.shape},  and labels, {labels.shape}, have different shapes."
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
        #running_test_loss /= len(test_ds)
        #writer.add_scalar("Loss/test", running_test_loss , epoch)

        if ep % cfg.train.log_freq == 0:
            logger.info(f"Epoch {ep:3>} [train, val, test]: [{running_loss:.6f}, {running_val_loss:.6f}, {running_test_loss:.6f}], Early stop count = {early_stop_count}")
            #logger.info(f"Epoch {ep:3>} train loss: {running_loss:.6f}")
            #logger.info(f"Epoch {ep:3>} validation loss  : {running_val_loss:.6f}")
            #logger.info(f"Epoch {ep:3>} test loss  : {running_test_loss:.6f}")
            #logger.info(f"Early stop count = {early_stop_count}")
    # save model
    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".pt")
    torch.save(model.state_dict(), filename)
    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".yaml")
    cfg.to_yaml(filename)
    logger.info("Finished training!")




if __name__ == "__main__":
    train3()
