import importlib
import sys
import logging
import torch
from torch.utils.data import random_split
sys.path.append("")

from GPTime.config import cfg
from GPTime.model.data import DeterministicTSDataSet

logger = logging.getLogger(__name__)

Criterion = getattr(importlib.import_module(cfg.train.criterion_module), cfg.train.criterion_name)
Optimizer = getattr(importlib.import_module(cfg.train.optimizer_module), cfg.train.optimizer_name)
Model = getattr(importlib.import_module(cfg.train.model_module), cfg.train.model_name)
Dataset = getattr(importlib.import_module(cfg.dataset.dataset_module), cfg.dataset.dataset_name)
DataLoader = getattr(importlib.import_module(cfg.train.dataloader_module), cfg.train.dataloader_name)



def setup():
    if Model.__name__ == "MLP":
        model_params = cfg.train.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = cfg.train.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = cfg.train.model_params_tcn
    else:
        logger.warning("Unknown model name.")
    model = Model(**model_params)
    criterion = Criterion(**cfg.train.criterion_params)
    optimizer = Optimizer(model.parameters(), **cfg.train.optimizer_params)
    
    # Dataset
    ds = Dataset(
        memory=model.memory,
        convolutions=True if Model.__name__=="TCN" else False,
        **cfg.dataset.dataset_params
        )
    train_length = int(ds.__len__() * cfg.train.train_set_size)
    val_length = int(ds.__len__() * cfg.train.val_set_size)
    test_length = ds.__len__() - train_length - val_length
    train_ds, ds_val_tmp, ds_test_tmp  = random_split(
        dataset=ds, 
        lengths=[train_length, val_length, test_length],
        generator=torch.torch.Generator()
        )
    val_ds = DeterministicTSDataSet(ds_val_tmp, num_tests_per_ts=cfg.train.num_tests_per_ts)
    test_ds = DeterministicTSDataSet(ds_test_tmp, num_tests_per_ts=cfg.train.num_tests_per_ts)

    # Dataloader
    train_dl = DataLoader(train_ds, **cfg.train.dataloader_params)
    val_dl = DataLoader(val_ds, **cfg.train.dataloader_params)
    test_dl = DataLoader(train_ds, **cfg.train.dataloader_params)

    return model, optimizer, criterion, train_dl, val_dl, test_dl


def epoch():
    pass

def eval():
    pass

def test()


def train():
    model, optimizer, criterion, train_dl, val_dl, test_dl = setup()



def train():
    if Model.__name__ == "MLP":
        model_params = cfg.train.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = cfg.train.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = cfg.train.model_params_tcn
    else:
        logger.warning("Unknown model name.")
    model = Model(**model_params)
    criterion = Criterion(**cfg.train.criterion_params)
    optimizer = Optimizer(model.parameters(), **cfg.train.optimizer_params)
    
    # Dataset
    ds = Dataset(
        memory=model.memory,
        convolutions=True if Model.__name__=="TCN" else False,
        **cfg.dataset.dataset_params
        )
    train_length = int(ds.__len__() * cfg.train.train_set_size)
    val_length = int(ds.__len__() * cfg.train.val_set_size)
    test_length = ds.__len__() - train_length - val_length
    train_ds, ds_val_tmp, ds_test_tmp  = random_split(
        dataset=ds, 
        lengths=[train_length, val_length, test_length],
        generator=torch.torch.Generator()
        )
    val_ds = DeterministicTSDataSet(ds_val_tmp, num_tests_per_ts=cfg.train.num_tests_per_ts)
    test_ds = DeterministicTSDataSet(ds_test_tmp, num_tests_per_ts=cfg.train.num_tests_per_ts)

    # Dataloader
    train_dl = DataLoader(train_ds, **cfg.train.dataloader_params)
    val_dl = DataLoader(val_ds, **cfg.train.dataloader_params)
    test_dl = DataLoader(train_ds, **cfg.train.dataloader_params)

    # epoch
    for epoch in range(cfg.train.max_epochs):

        running_loss = 0.0
        for i, data in enumerate(train_dl):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            # early stop
        
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        # add some tensorbard

    # save model


            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    
    # eval

    # save model



if __name__ == "__main__":
    train()