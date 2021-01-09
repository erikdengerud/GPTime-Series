import importlib
import logging
import os
import torch
import sys

sys.path.append("")

from GPTime.config import cfg
from GPTime.utils.scoring import predict_M4, score_M4

logger = logging.getLogger(__name__)

def evaluate(evaluate_cfg):
    Model = getattr(importlib.import_module(evaluate_cfg.model_module), evaluate_cfg.model_name)
    # load model
    if Model.__name__ == "MLP":
        model_params = evaluate_cfg.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = evaluate_cfg.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = evaluate_cfg.model_params_tcn
    else:
        logger.warning("Unknown model name.")
    model = Model(**model_params).double()
    model_path = os.path.join(evaluate_cfg.model_save_path, evaluate_cfg.name + ".pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # predict test data
    # TODO: Preds should take in validation or test. Can't snoop the test data when selecting models etc.
    preds, df_preds = predict_M4(model=model, scale=evaluate_cfg.scale, seasonal_init=evaluate_cfg.seasonal_init, val_set=evaluate_cfg.val_set)

    # score test data, save scores
    result_file = os.path.join(evaluate_cfg.results_path, evaluate_cfg.name + ".csv")
    d = score_M4(preds, df_results_name=result_file, val=evaluate_cfg.val_set)
    logger.info(d)
    # save predictions? YES!
    csv_path = os.path.join(evaluate_cfg.predictions_path, evaluate_cfg.name + ".csv")
    df_preds.to_csv(csv_path)
    # return d


if __name__ == "__main__":
    evaluate()