import importlib
import logging
import os
import torch
import sys
sys.path.append("")

from GPTime.config import cfg
from GPTime.utils.scoring import predict_M4, score_M4

logger = logging.getLogger(__name__)

Criterion = getattr(importlib.import_module(cfg.train.criterion_module), cfg.train.criterion_name)
Optimizer = getattr(importlib.import_module(cfg.train.optimizer_module), cfg.train.optimizer_name)
Model = getattr(importlib.import_module(cfg.train.model_module), cfg.train.model_name)
Dataset = getattr(importlib.import_module(cfg.dataset.dataset_module), cfg.dataset.dataset_name)
DataLoader = getattr(importlib.import_module(cfg.train.dataloader_module), cfg.train.dataloader_name)

def evaluate(box_config):

    # load model
    if Model.__name__ == "MLP":
        model_params = box_config.train.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = box_config.train.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = box_config.train.model_params_tcn
    else:
        logger.warning("Unknown model name.")   
    model = Model(**model_params).double()
    model_path = os.path.join(box_config.train.model_save_path, box_config.run.name + ".pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # predict test data
    preds = predict_M4(model=model)

    # score test data, save scores
    result_file = os.path.join(box_config.evaluate.results_path, box_config.run.name +".csv")
    d = score_M4(preds, df_results_name=result_file)

    # save predictions?
    #return d

if __name__ == "__main__":
    evaluate(cfg)