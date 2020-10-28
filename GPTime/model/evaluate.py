import importlib
import logging
import os
import torch
import sys
sys.path.append("")

from GPTime.config import cfg
from GPTime.utils.scoring2 import predict_M4, score_M4

logger = logging.getLogger(__name__)

Criterion = getattr(importlib.import_module(cfg.train.criterion_module), cfg.train.criterion_name)
Optimizer = getattr(importlib.import_module(cfg.train.optimizer_module), cfg.train.optimizer_name)
Model = getattr(importlib.import_module(cfg.train.model_module), cfg.train.model_name)
Dataset = getattr(importlib.import_module(cfg.dataset.dataset_module), cfg.dataset.dataset_name)
DataLoader = getattr(importlib.import_module(cfg.train.dataloader_module), cfg.train.dataloader_name)

def evaluate():

    # load model
    if Model.__name__ == "MLP":
        model_params = cfg.train.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = cfg.train.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = cfg.train.model_params_tcn
    else:
        logger.warning("Unknown model name.")   
    model = Model(**model_params).double()
    model_path = os.path.join(cfg.train.model_save_path, cfg.run.name + ".pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # predict test data
    preds = predict_M4(model=model)

    # score test data, save scores
    result_file = os.path.join(cfg.evaluate.results_path, cfg.run.name +".csv")
    d = score_M4(preds, df_results_name=result_file)
    logger.info(d)
    # save predictions?
    #return d

if __name__ == "__main__":
    evaluate()