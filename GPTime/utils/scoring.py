"""
Calculating the overall M4 score for a model
"""
import torch.nn as nn
import torch

import pandas as pd
import glob
import logging
import numpy as np
from typing import Dict, List, Tuple

from metrics import MASE, SMAPE, OWA

import sys

sys.path.append("")

logger = logging.getLogger(__name__)

from GPTime.config import cfg


def multi_step_predict(
    model: nn.Module,
    train_data: np.array,
    horizon: int,
    mase_scale: bool,
    scale: pd.DataFrame,
) -> np.array:

    if mase_scale:
        df_memory = pd.DataFrame(train_data)
        assert len(scale) == len(df_memory)
        train_data = df_memory.div(scale, axis=0).values

    x = torch.from_numpy(train_data)
    with torch.no_grad():
        for h in range(horizon):
            current_x = x[:, -model.memory :]
            pred = model.forward(current_x)
            x = torch.cat((x, pred), axis=1)
    predictions = x[:, -horizon:].cpu().detach().numpy()

    # Descale predictions
    if mase_scale:
        df_predictions = pd.DataFrame(predictions)
        assert len(scale) == len(df_predictions)
        predictions = df_predictions.mul(scale, axis=0).values
    return predictions


def period_season_file(fname: str, period_dict: Dict) -> Tuple[int, str]:
    for p in period_dict.keys():
        if p.lower() in fname.lower():
            return period_dict[p], p
    logger.warning(f"No season or period found for file {fname}.")
    return 1, ""


def create_training_data(fname: str, memory: int, season: int) -> np.array:
    frequency_tmp = []
    try:
        df = pd.read_csv(fname, index_col=0)
    except Exception as e:
        logger.warning(f"Could not read file: {fname}.")
        raise

    Y = df.to_numpy()
    # create training data as long as the memory of the model. This is a for loop due to the format of the .csv.
    for i in range(Y.shape[0]):
        ts = Y[i][~np.isnan(Y[i])]
        if len(ts) < model.memory:
            # pad with token. 0 for now.
            frequency_tmp.append(
                np.pad(
                    ts,
                    pad_width=(model.memory - len(s), 0),
                    mode="constant",
                    constant_values=0,
                )
            )
        else:
            frequency_tmp.append(ts[-model.memory :])

    mase_scale = (
        df.diff(periods=season, axis=1).abs().mean(axis=1).reset_index(drop=True)
    )

    return frequency_tmp, mase_scale


def predict_M4(model: nn.Module) -> List:
    """ Predicting M4 using a model provided. """
    assert hasattr(model, "forward")
    assert hasattr(model, "memory")
    model.eval()

    all_train_files = glob.glob(cfg.path.m4_train + "*")

    all_predictions = []
    for fname in all_train_files:
        season, period = period_season_file(
            fname=fname, period_dict=cfg.scoring.m4.periods
        )
        logger.info(f"Predicting {period}")
        frequency_train_data, scale = create_training_data(
            fname=fname, memory=model.memory, season=season
        )

        predictions = multi_step_predict(
            model=model,
            horizon=cfg.scoring.m4.horizons[period],
            train_data=frequency_train_data,
            mase_scale=cfg.scoring.m4.scale_mase,
            scale=scale,
        )

        for prediction in predictions:
            all_predictions.append(prediction)

    return all_predictions


def score_M4(predictions: List) -> Tuple[Dict]:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    metrics = {}
    frequency_metrics = {}
    for metric in cfg.scoring.metrics.keys():
        if cfg.scoring.metrics[metric]:
            metrics[metric] = []
            frequency_metrics[metric] = []
    # Read in and prepare the data
    all_test_files = glob.glob(cfg.path.m4_test + "*")
    all_train_files = glob.glob(cfg.path.m4_test + "*")
    crt_pred_index = 0
    tot_mase = 0
    tot_smape = 0
    for fname_train, fname_test in zip(all_train_files, all_test_files):
        # logger.info(fname_test)
        df_train = pd.read_csv(fname_train, index_col=0)
        df_test = pd.read_csv(fname_test, index_col=0)

        season, period = period_season_file(
            fname=fname_train, period_dict=cfg.scoring.m4.periods
        )
        horizon = cfg.scoring.m4.horizons[period]
        scale = (
            df_train.diff(periods=season, axis=1)
            .abs()
            .mean(axis=1)
            .reset_index(drop=True)
        )

        Y = df_test.values[:, :horizon]
        # logger.info(f"horizon : {horizon}")
        # logger.info(f"df_test.isna().sum() : {df_test.isna().sum()}")
        # logger.info(f"crt_pred_index: {crt_pred_index}")
        # logger.info(f"Y.shape[0]: {Y.shape[0]}")
        index = crt_pred_index + Y.shape[0]
        # logger.info(type(index))
        predicted = np.array(predictions[crt_pred_index:index])
        # logger.info(f"np.sum(np.isnan(Y)): {np.sum(np.isnan(Y))}")
        assert np.sum(np.isnan(Y)) == 0
        assert Y.shape == predicted.shape
        # logger.info(f"Y.shape: {Y.shape}")
        # logger.info(f"predicted.shape: {predicted.shape}")

        mase_freq = MASE(Y, predicted, scale)
        smape_freq = SMAPE(Y, predicted)
        # logger.info(period)
        tot_mase += mase_freq * Y.shape[0]
        tot_smape += smape_freq * Y.shape[0]
        logger.info(f"{period:<9} MASE : {mase_freq}")
        logger.info(f"{period:<9} SMAPE: {smape_freq}")

        crt_pred_index += Y.shape[0]

    tot_mase = tot_mase / crt_pred_index
    tot_smape = tot_smape / crt_pred_index
    logger.info(f"TOTAL MASE : {tot_mase}")
    logger.info(f"TOTAL SMAPE: {tot_smape}")
    tot_owa = OWA(tot_mase, tot_smape)
    logger.info(f"TOTAL OWA: {tot_owa}")

    return tot_mase, tot_smape, tot_owa


if __name__ == "__main__":
    # Dummy model:
    import torch.nn as nn
    import torch.nn.functional as F

    class MLP(nn.Module):
        def __init__(self, in_features=10, out_features=1, n_hidden=16):
            super(MLP, self).__init__()
            self.l1 = nn.Linear(in_features=in_features, out_features=n_hidden)
            self.l2 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
            self.out = nn.Linear(in_features=n_hidden, out_features=out_features)
            self.memory = in_features

        def forward(self, x):
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            out = self.out(x)
            return out

    model = MLP(in_features=10, out_features=1, n_hidden=16).double()

    # predict
    preds = predict_M4(model=model)
    # logger.info(len(preds))
    score_M4(preds)

    # score


#### BELOW TODO ###
def plot_predictions():
    return 0


def plot_importance():
    return 0


def predict_electricity(model: nn.Module) -> List:
    """ Predicting M4 using a model provided. """

    return 0


def score_electricity(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return 0


def predict_traffic(model: nn.Module) -> List:
    """ Predicting M4 using a model provided. """

    return 0


def score_traffic(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return 0


def predict_tourism(model: nn.Module) -> List:
    """ Predicting M4 using a model provided. """

    return 0


def score_tourism(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return 0


def predict_M3(model: nn.Module) -> List:
    """ Predicting M4 using a model provided. """

    return 0


def score_M3(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return 0


def predict_wiki(model: nn.Module) -> List:
    """ Predicting M4 using a model provided. """

    return 0


def score_wiki(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return 0