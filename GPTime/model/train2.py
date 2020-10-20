import importlib
import logging
import torch
import os
import numpy as np
import sys
sys.path.append("")

from GPTime.config import cfg
from GPTime.model.data import DeterministicTSDataSet

logger = logging.getLogger(__name__)
    
Model = getattr(importlib.import_module(cfg.train.model_module), cfg.train.model_name)
Criterion = getattr(importlib.import_module(cfg.train.criterion_module), cfg.train.criterion_name)
Optimizer = getattr(importlib.import_module(cfg.train.optimizer_module), cfg.train.optimizer_name)
Dataset = getattr(importlib.import_module(cfg.dataset.dataset_module), cfg.dataset.dataset_name)
DataLoader = getattr(importlib.import_module(cfg.train.dataloader_module), cfg.train.dataloader_name)

def train2():
    X_train = np.load(cfg.dataset.dataset_params.dataset_paths.M4_global)
    X_test = np.load(cfg.dataset.dataset_params.dataset_paths.M4_global_test)

    train = X_train[: int(X_train.shape[0] * 0.85), :]
    val = X_train[int(X_train.shape[0] * 0.85) :, :]

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

    ds = Dataset(
        arr=train, 
        memory=model.memory, 
        convolutions=True if Model.__name__=="TCN" else False,
        **cfg.dataset.dataset_params
        )
    ds_val = Dataset(
        arr=val, 
        memory=model.memory, 
        convolutions=True if Model.__name__=="TCN" else False,
        **cfg.dataset.dataset_params
        )
        
    trainloader = DataLoader(ds, **cfg.train.dataloader_params)
    valloader = DataLoader(ds_val, **cfg.train.dataloader_params)

    num_epochs = 1000
    tenacity = 25
    early_stop_count = 0
    low_loss = np.inf
    for ep in range(1, cfg.train.max_epochs + 1):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0], data[1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, data in enumerate(valloader, 0):
                inputs, labels = data[0], data[1]
                outputs = model(inputs)
                loss = criterion(outputs.flatten(), labels)
                val_loss += loss.item()
            if val_loss < low_loss:
                early_stop_count = 0
                low_loss = val_loss
            else:
                early_stop_count += 1
            if early_stop_count > tenacity:
                print(f"Early stop after epoch {ep}.")
                break
            # print(f"Epoch {epoch:3>} loss: {running_loss}")
        if ep % 20 == 0:
            logger.info(f"Epoch {ep:3>} train loss: {running_loss}")
            logger.info(f"Epoch {ep:3>} val loss  : {val_loss}")
            logger.info(f"Early stop count = {early_stop_count}")
    logger.info("Finished training!")

    with torch.no_grad():
        Y_hat = []
        for i in range(6):  # X_test.shape[1]
            if i == 0:
                X = X_train[:, i + 1 :]
            else:
                X = np.concatenate(
                    (X_train[:, (i + 1) :], X_test[:, :i]), axis=1
                )
            sample = torch.from_numpy(X)
            out = model(sample).cpu().detach().numpy().flatten()
            Y_hat.append(out)

    forecast = np.stack(Y_hat, axis=1)
    # calculate mase (mae since we have already scaled)
    error = np.mean(np.abs(forecast - X_test))
    print(f"MASE : {error}")
    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".pt")
    torch.save(model.state_dict(), filename)
    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".yaml")
    cfg.to_yaml(filename)

if __name__ == "__main__":
    train2()
