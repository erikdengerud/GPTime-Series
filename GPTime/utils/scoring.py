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
import sys

sys.path.append("")

logger = logging.getLogger(__name__)

from GPTime.config import cfg
from GPTime.utils.metrics import MASE, SMAPE, OWA


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
            current_x = x[:, -getattr(model, "memory") :]
            pred = getattr(model, "forward")(current_x)
            x = torch.cat((x, pred), axis=1)
    predictions = x[:, -horizon:].cpu().detach().numpy()

    # Descale predictions
    if mase_scale:
        df_predictions = pd.DataFrame(predictions)
        assert len(scale) == len(df_predictions)
        predictions = df_predictions.mul(scale, axis=0).values
    return predictions


def period_num_str_file(fname: str, period_dict: Dict) -> Tuple[int, str]:
    for p in period_dict.keys():
        if p.lower() in fname.lower():
            return period_dict[p], p
    logger.warning(f"No season or period found for file {fname}.")
    return 1, ""


def create_training_data(fname: str, memory: int, period_num: int) -> Tuple:
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
        if len(ts) < memory:
            # pad with token. 0 for now.
            frequency_tmp.append(
                np.pad(
                    ts,
                    pad_width=(memory - len(ts), 0),
                    mode="constant",
                    constant_values=0,
                )
            )
        else:
            frequency_tmp.append(ts[-memory:])

    mase_scale = (
        df.diff(periods=period_num, axis=1).abs().mean(axis=1).reset_index(drop=True)
    )

    return frequency_tmp, mase_scale


def predict_M4(model: nn.Module) -> np.array:
    """ Predicting M4 using a model provided. """
    assert hasattr(model, "forward")
    assert hasattr(model, "memory")
    model.eval()

    all_train_files = glob.glob(cfg.path.m4_train + "*")
    assert len(all_train_files) > 0, f"Did not find data in {cfg.path.m4_train}"
    frames = []
    for fname in all_train_files:
        period_num, period_str = period_num_str_file(
            fname=fname, period_dict=cfg.scoring.m4.periods
        )
        frequency_train_data, scale = create_training_data(
            fname=fname, memory=getattr(model, "memory"), period_num=period_num
        )
        predictions = multi_step_predict(
            model=model,
            horizon=cfg.scoring.m4.horizons[period_str],
            train_data=frequency_train_data,
            mase_scale=cfg.scoring.m4.scale_mase,
            scale=scale,
        )
        df = pd.DataFrame(predictions)
        frames.append(df)

    df_all = pd.concat(frames)
    predictions = df_all.values

    return df_all.values


def score_M4(
    predictions: np.array, df_results_name: str = "GPTime/results/M4/test.csv"
) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    """
    metrics = {}
    frequency_metrics = {}
    for metric in cfg.scoring.metrics.keys():
        if cfg.scoring.metrics[metric]:
            metrics[metric] = []
            frequency_metrics[metric] = []
    """
    frequency_metrics: Dict[str, Dict[str, float]] = {}
    # Read in and prepare the data
    all_test_files = glob.glob(cfg.path.m4_test + "*")
    all_train_files = glob.glob(cfg.path.m4_test + "*")
    crt_pred_index = 0
    tot_mase = 0.0
    tot_smape = 0.0
    for fname_train, fname_test in zip(all_train_files, all_test_files):
        df_train = pd.read_csv(fname_train, index_col=0)
        df_test = pd.read_csv(fname_test, index_col=0)

        period_num, period_str = period_num_str_file(
            fname=fname_train, period_dict=cfg.scoring.m4.periods
        )
        horizon = cfg.scoring.m4.horizons[period_str]
        scale = (
            df_train.diff(periods=period_num, axis=1)
            .abs()
            .mean(axis=1)
            .reset_index(drop=True)
        ).values

        Y = df_test.values[:, :horizon]
        index = crt_pred_index + Y.shape[0]
        predicted = predictions[crt_pred_index:index, :horizon]

        assert np.sum(np.isnan(Y)) == 0, "NaNs in Y"
        assert np.sum(np.isnan(predicted)) == 0, "NaNs in predictions"
        assert Y.shape == predicted.shape, "Y and predicted have different shapes"

        mase_freq = MASE(Y, predicted, scale)
        smape_freq = SMAPE(Y, predicted)
        owa_freq = OWA(mase=mase_freq, smape=smape_freq, freq=period_str)
        tot_mase += mase_freq * Y.shape[0]
        tot_smape += smape_freq * Y.shape[0]

        frequency_metrics[period_str] = {}
        frequency_metrics[period_str]["MASE"] = mase_freq
        frequency_metrics[period_str]["SMAPE"] = smape_freq
        frequency_metrics[period_str]["OWA"] = owa_freq

        crt_pred_index += Y.shape[0]

    tot_mase = tot_mase / crt_pred_index
    tot_smape = tot_smape / crt_pred_index
    tot_owa = OWA(tot_mase, tot_smape, freq="global")

    frequency_metrics["GLOBAL"] = {}
    frequency_metrics["GLOBAL"]["MASE"] = tot_mase
    frequency_metrics["GLOBAL"]["SMAPE"] = tot_smape
    frequency_metrics["GLOBAL"]["OWA"] = tot_owa

    df = pd.DataFrame(frequency_metrics).T
    df.to_csv(df_results_name)

    return frequency_metrics


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

    mlp = MLP(in_features=10, out_features=1, n_hidden=16).double()
    # predict
    preds = predict_M4(model=mlp)
    logger.info(len(preds))
    d = score_M4(preds)
    print(d)
    df = pd.DataFrame(d).T
    logger.debug(df.head(10))
    logger.debug(df.shape)

    # score


#### BELOW TODO ###
def plot_predictions():
    return 0


def plot_importance():
    return 0


def predict_electricity(model: nn.Module) -> Dict:
    """ Predicting M4 using a model provided. """

    return {}


def score_electricity(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return {}


def predict_traffic(model: nn.Module) -> Dict:
    """ Predicting M4 using a model provided. """

    return {}


def score_traffic(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return {}


def predict_tourism(model: nn.Module) -> Dict:
    """ Predicting M4 using a model provided. """

    return {}


def score_tourism(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return {}


def predict_M3(model: nn.Module) -> Dict:
    """ Predicting M4 using a model provided. """

    return {}


def score_M3(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return {}


def predict_wiki(model: nn.Module) -> Dict:
    """ Predicting M4 using a model provided. """

    return {}


def score_wiki(actual: List, predictions: List) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    return {}