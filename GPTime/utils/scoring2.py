"""
Calculating the overall M4 score for a model
"""
import torch.nn as nn
import torch
import importlib
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

Scaler = getattr(
    importlib.import_module(cfg.train.scaler_module), cfg.train.scaler_name
)


def period_from_fname(fname: str, period_dict: Dict) -> Tuple[int, str]:
    """
    Get the periodicity from a filename as integer and string.
    """
    for p in period_dict.keys():
        if p.lower() in fname.lower():
            return period_dict[p], p
    logger.warning(f"No season or period found for file {fname}.")
    return 1, ""


def create_training_data(fname: str) -> np.array:
    """
    Reverting the training data.
    The data is in a form where the first value is in the first column etc. and it is
    padded at the end with nans to get the same length for all ts. This function changes
    the format s.t. the last column is the first value etc.
    """
    Y = pd.read_csv(fname, index_col=0).to_numpy()
    tmp = []
    for i in range(Y.shape[0]):
        ts = Y[i][~np.isnan(Y[i])]
        tmp.append(
            np.pad(
                ts,
                pad_width=(Y.shape[1] - len(ts), 0),
                mode="constant",
                constant_values=np.nan,
            )
        )
    X = np.array(tmp)
    assert X.shape == Y.shape, f"diff in shapes; Y.shape:{Y.shape}, X.shape:{X.shape}"
    return X


def multi_step_predict(
    model: nn.Module, train_data: np.array, horizon: int
) -> np.array:
    """
    Multi step forecasting with a model on training data.
    """
    memory = getattr(model, "memory")
    with torch.no_grad():
        for i in range(horizon):
            sample = torch.from_numpy(train_data[:, -memory:])
            out = model(sample).cpu().detach().numpy()
            train_data = np.hstack((train_data, out))
    forecast = train_data[:, -horizon:]
    return forecast


def predict_M4(model: nn.Module) -> np.array:
    """ Predicting M4 using a model provided. """
    assert hasattr(model, "forward")
    assert hasattr(model, "memory")
    model.eval()

    all_train_files = glob.glob(cfg.path.m4_train + "*")
    assert len(all_train_files) > 0, f"Did not find data in {cfg.path.m4_train}"
    frames = []
    for fname in all_train_files:
        period_numeric, period_str = period_from_fname(
            fname=fname, period_dict=cfg.scoring.m4.periods
        )
        X = create_training_data(fname=fname)
        scaler = Scaler().fit(X, freq=period_numeric)
        X = scaler.transform(X)
        # check if this is the same as train2 version ^^
        # check if vv is similar predictions for set input.
        predictions = multi_step_predict(
            model=model,
            train_data=X,
            horizon=cfg.scoring.m4.horizons[period_str],
        )
        predictions = scaler.inverse_transform(predictions)

        df = pd.DataFrame(predictions)
        logger.debug(df.shape)
        frames.append(df)

    df_all = pd.concat(frames)
    predictions = df_all.values
    logger.info(f"predictions shape: {predictions.shape}")
    # logger.info(df_all.tail())
    # logger.info(df_all.iloc[0])
    return df_all.values


def score_M4(
    predictions: np.array, df_results_name: str = "GPTime/results/M4/test.csv"
) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    frequency_metrics: Dict[str, Dict[str, float]] = {}
    # Read in and prepare the data
    all_test_files = glob.glob(cfg.path.m4_test + "*")
    all_train_files = glob.glob(cfg.path.m4_train + "*")
    all_test_files.sort()
    all_train_files.sort()
    crt_pred_index = 0
    tot_mase = 0.0
    tot_smape = 0.0
    for fname_train, fname_test in zip(all_train_files, all_test_files):
        df_train = pd.read_csv(fname_train, index_col=0)
        df_test = pd.read_csv(fname_test, index_col=0)

        period_num, period_str = period_from_fname(
            fname=fname_train, period_dict=cfg.scoring.m4.periods
        )
        horizon = cfg.scoring.m4.horizons[period_str]

        scale = Scaler().fit(df_train.values, freq=period_num).scale_.flatten()
        # logger.info(f"scale.shape: {scale.shape}")
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
