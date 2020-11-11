import importlib
import logging
import torch
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


def train2():
    X_train = np.load(cfg.dataset.dataset_params.dataset_paths.M4_global)
    X_test = np.load(cfg.dataset.dataset_params.dataset_paths.M4_global_test)

    train = X_train[: int(X_train.shape[0] * 0.5), :]
    val = X_train[int(X_train.shape[0] * 0.5) :, :]

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

    ds = Dataset(
        arr=train,
        memory=model.memory,
        convolutions=True if Model.__name__ == "TCN" else False,
        **cfg.dataset.dataset_params,
    )
    ds_val = Dataset(
        arr=val,
        memory=model.memory,
        convolutions=True if Model.__name__ == "TCN" else False,
        **cfg.dataset.dataset_params,
    )

    trainloader = DataLoader(ds, **cfg.train.dataloader_params)
    valloader = DataLoader(ds_val, **cfg.train.dataloader_params)

    tenacity =5 
    early_stop_count = 0
    low_loss = np.inf
    np.random.seed(1729)
    torch.manual_seed(1729)
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
        if ep % 10 == 0:
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
                X = np.concatenate((X_train[:, (i + 1) :], X_test[:, :i]), axis=1)
            sample = torch.from_numpy(X)
            out = model(sample).cpu().detach().numpy().flatten()
            Y_hat.append(out)

    forecast = np.stack(Y_hat, axis=1)
    # calculate mase (mae since we have already scaled)
    error = np.mean(np.abs(forecast - X_test))
    error_axis = np.mean(np.mean(np.abs(forecast - X_test), axis=1))
    error_yearly = np.mean(np.abs(forecast[-23000:] - X_test[-23000:]))
    error_yearly_axis = np.mean(
        np.mean(np.abs(forecast[-23000:] - X_test[-23000:]), axis=1)
    )
    logger.info(f"MASE : {error}")
    logger.info(f"MASE axis : {error_axis}")
    logger.info(f"MASE yearly : {error_yearly}")
    logger.info(f"MASE axis yearly : {error_yearly_axis}")

    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".pt")
    torch.save(model.state_dict(), filename)
    filename = os.path.join(cfg.train.model_save_path, cfg.run.name + ".yaml")
    cfg.to_yaml(filename)

    # Testing yearly data recursive
    df_yearly_train = pd.read_csv(
        "GPTime/data/raw/M4/M4train/Yearly-train.csv", index_col=0
    )
    df_yearly_test = pd.read_csv(
        "GPTime/data/raw/M4/M4test/Yearly-test.csv", index_col=0
    )
    scale = (
        df_yearly_train.diff(periods=1, axis=1)
        .abs()
        .mean(axis=1)
        .reset_index(drop=True)
    )

    X_train_yearly = df_yearly_train.div(scale.values, axis=0).to_numpy()
    X_test_yearly = df_yearly_test.div(scale.values, axis=0).to_numpy()

    # X_train_yearly = df_yearly_train.to_numpy()
    # X_test_yearly = df_yearly_test.to_numpy()
    ts_train = []
    ts_test = []
    for i in range(X_train_yearly.shape[0]):
        s_train = X_train_yearly[i][~np.isnan(X_train_yearly[i])]
        s_test = X_test_yearly[i][~np.isnan(X_test_yearly[i])]
        ts_train.append(s_train[-12:])  # shortest in the train set
        ts_test.append(s_test[:6])  # shortest in the test set

    df_train_out = pd.DataFrame(ts_train)
    df_test_out = pd.DataFrame(ts_test)

    X_train = np.array(ts_train)
    X_test = np.array(ts_test)

    logger.info("recursive forecasting Yearly")

    with torch.no_grad():
        for i in range(6):  # X_test.shape[1]
            sample = torch.from_numpy(X_train[:, -12:])
            out = model(sample).cpu().detach().numpy()
            X_train = np.hstack((X_train, out))

    forecast = X_train[:, -6:]

    error = np.mean(np.abs(forecast - X_test))
    error_axis = np.mean(np.mean(np.abs(forecast - X_test), axis=1))
    logger.info(f"MASE Yearly recursive: {error}")
    logger.info(f"MASE Yearly axis recursive: {error_axis}")

    X_train = np.array(ts_train)
    X_test = np.array(ts_test)
    logger.info(X_train.shape)
    logger.info(X_test.shape)
    logger.info("forecasting Yearly")
    with torch.no_grad():
        Y_hat = []
        for i in range(6):  # X_test.shape[1]
            if i == 0:
                X = X_train[:, i:]
            else:
                X = np.concatenate((X_train[:, i:], X_test[:, :i]), axis=1)
            sample = torch.from_numpy(X)
            out = model(sample).cpu().detach().numpy().flatten()
            Y_hat.append(out)

    forecast = np.stack(Y_hat, axis=1)

    error = np.mean(np.abs(forecast - X_test))
    logger.info(f"MASE Yearly one-step: {error}")
    error_axis = np.mean(np.mean(np.abs(forecast - X_test), axis=1))
    logger.info(f"MASE Yearly_axis one-step: {error_axis}")


if __name__ == "__main__":
    train2()
