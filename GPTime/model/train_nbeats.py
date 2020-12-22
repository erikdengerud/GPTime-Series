"""
Timeseries sampler
"""
import numpy as np
import pandas as pd
from typing import Iterator

from collections import OrderedDict

import numpy as np
import torch as t
from torch import optim
from GPTime.model.data_nbeats import M4Dataset, M4Meta
from GPTime.utils.scoring import score_M4, predict_M4
from GPTime.networks.mlp import MLP as my_MLP
from GPTime.networks.mlp import MLP_orig as my_MLP_orig
from GPTime.model.data import TSDataset
from GPTime.config import cfg

from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def mase(forecast, insample, outsample, frequency):
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))

def smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom

class MLP(t.nn.Module): 
    """
    MLP 
    """ 
    def __init__(self, input_size, output_size, layer_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = 10
        self.layer_size = layer_size
        self.frequency = 12
        self.memory = input_size

        self.layers = t.nn.ModuleList()
        self.layers.append(t.nn.Linear(in_features=input_size, out_features=layer_size))
        for i in range(1, self.n_layers-1):
            self.layers.append(t.nn.Linear(in_features=layer_size, out_features=layer_size))
        self.out_layer = t.nn.Linear(in_features=layer_size, out_features=output_size)

        print(f"Building model with frequency {self.frequency}")
        print(f"Input size of model: {input_size}")

    def forward(self, x, mask, freq):
        naive = x[:, -1].unsqueeze(1)
        for i, layer in enumerate(self.layers):
            x = t.relu(layer(x))
        block_forecast = self.out_layer(x)
        forecast = naive  + block_forecast
        return forecast


class TimeseriesSampler:
    def __init__(self, timeseries: np.ndarray):
        self.timeseries = [ts for ts in timeseries]
        self.window_sampling_limit = 27
        self.batch_size = 1024
        self.insample_size = 72
        self.outsample_size = 1

    def __iter__(self):
        while True:
            insample = np.zeros((self.batch_size, self.insample_size))
            insample_mask = np.zeros((self.batch_size, self.insample_size))
            outsample = np.zeros((self.batch_size, self.outsample_size))
            outsample_mask = np.zeros((self.batch_size, self.outsample_size))
            sampled_ts_indices = np.random.randint(len(self.timeseries), size=self.batch_size)
            for i, sampled_index in enumerate(sampled_ts_indices):
                sampled_timeseries = self.timeseries[sampled_index]
                cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                              high=len(sampled_timeseries),
                                              size=1)[0]

                insample_window = sampled_timeseries[max(0, cut_point - self.insample_size):cut_point]
                insample[i, -len(insample_window):] = insample_window
                insample_mask[i, -len(insample_window):] = 1.0
                outsample_window = sampled_timeseries[
                                   cut_point:min(len(sampled_timeseries), cut_point + self.outsample_size)]
                outsample[i, :len(outsample_window)] = outsample_window
                outsample_mask[i, :len(outsample_window)] = 1.0
            yield insample, insample_mask, outsample, outsample_mask

    def last_insample_window(self):
        insample = np.zeros((len(self.timeseries), self.insample_size))
        insample_mask = np.zeros((len(self.timeseries), self.insample_size))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.insample_size:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]], dtype=object)


def mase_loss(insample: t.Tensor, freq: int, forecast: t.Tensor, target: t.Tensor, mask: t.Tensor):
    masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
    masked_masep_inv = divide_no_nan(mask, masep[:, None])
    return t.mean(t.abs(target - forecast) * masked_masep_inv)

def default_device() -> t.device:
    return t.device('cuda' if t.cuda.is_available() else 'cpu')

def to_tensor(array: np.ndarray) -> t.Tensor:
    return t.tensor(array, dtype=t.float32).to(default_device())

def divide_no_nan(a, b):
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

def summarize_groups(scores, test_set):
    scores_summary = OrderedDict()

    def group_count(group_name):
        return len(np.where(test_set.groups == group_name)[0])

    weighted_score = {}
    for g in ['Yearly', 'Quarterly', 'Monthly']:
        weighted_score[g] = scores[g] * group_count(g)
        scores_summary[g] = scores[g]

    others_score = 0
    others_count = 0
    for g in ['Weekly', 'Daily', 'Hourly']:
        others_score += scores[g] * group_count(g)
        others_count += group_count(g)
    weighted_score['Others'] = others_score
    scores_summary['Others'] = others_score / others_count

    average = np.sum(list(weighted_score.values())) / len(test_set.groups)
    scores_summary['Average'] = average

    return scores_summary

def main():
    print("Starting main")
    lookback = 4
    iterations = 15
    learning_rate = 0.001
    maxscale = True
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }

    dataset = M4Dataset.load(training=True)
    print("Loaded dataset.")
    print(f"Dataset size: {dataset.values.shape}")
    forecasts = []
    for seasonal_pattern in M4Meta.seasonal_patterns:
        if seasonal_pattern == "Monthly":
            print(f"Training model for seasonal pattern {seasonal_pattern}")
            history_size_in_horizons = history_size[seasonal_pattern]
            horizon = M4Meta.horizons_map[seasonal_pattern]
            input_size = lookback * horizon

            # Training Set
            training_values = group_values(dataset.values, dataset.groups, seasonal_pattern)

            ds = TSDataset(
                memory=72,
                convolutions=False,
                **cfg.dataset.dataset_params
                )
            train_loader = DataLoader(dataset=ds, batch_size=1024)

            print(training_values.shape)
            print(len(training_values))
            all_ts = [ts["values"] for ts in ds.all_ts if ts["frequency"] == "M"]
            training_set = TimeseriesSampler(timeseries=training_values)
            training_set_train = TimeseriesSampler(timeseries=all_ts)
            print(f"len(training_set.timeseries) = {len(training_set.timeseries)}")
            #iter_train = iter(training_set)
            iter_train = iter(training_set_train)

            # check if they are similar
            """
            print(training_values[0])
            print(all_ts[0])
            num_sim = 0
            for ts in training_values:
                for my_ts in all_ts:
                    if ts == my_ts:
                        num_sim += 1
            print(f"Num sim = {num_sim}")
            print(f"pct sim = {num_sim/len(all_ts) * 100:.2f}")
            """

            #model = MLP(input_size=72, output_size=1, layer_size=512)
            #model = my_MLP(in_features=72, out_features=1, num_layers=10, n_hidden=512)
            model = my_MLP_orig(in_features=72, out_features=1, num_layers=10, n_hidden=512).double()
            logger.info(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}") 
            # Train model
            print("Training model.")
            model = model.to(default_device())
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            lr_decay_step = iterations // 3
            if lr_decay_step == 0:
                lr_decay_step = 1

            running_loss = 0.0
            #for i in range(1, iterations + 1):
            for i, data in enumerate(train_loader):
                model.train()
                #x, x_mask, y, y_mask = map(to_tensor, next(iter_train))
                x, y, x_mask = data[0].to(default_device()), data[1].to(default_device()), data[2].to(default_device())
                y_mask = t.ones(y.shape).to(default_device())
                optimizer.zero_grad()
                if maxscale:
                    max_scale = t.transpose(t.max(x, 1).values.unsqueeze(0), 1, 0)                                                                                                                                     
                    x = t.div(x, max_scale)

                forecast = model(x, x_mask, 1)

                if maxscale:
                    forecast = t.mul(forecast, max_scale)

                training_loss = mase_loss(x, 12, forecast, y, y_mask)

                if np.isnan(float(training_loss)):
                    break

                training_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate * 0.5 ** (i // lr_decay_step)
                
                running_loss += training_loss.item()
                if (i+1) % 10 == 0:
                    print(f"Iter {i+1:<5d} [Avg. Loss, Loss] : [{running_loss / 500 :.4f}, {training_loss.item():.4f}]")
                    running_loss = 0.0
                
            # Build forecasts
            x, x_mask = map(to_tensor, training_set.last_insample_window())
            model.eval()
            with t.no_grad():
                x_np = x.cpu().detach().numpy()
                x_mask_np = x.cpu().detach().numpy()
                for i in range(horizon):
                    sample = t.from_numpy(x_np[:, -input_size:]).to(default_device()).double()
                    sample_mask = t.from_numpy(x_mask_np[:, -input_size:]).to(default_device()).double()
                    #logger.debug(f"sample_mask.shape: {sample_mask.shape}")
                    if maxscale:
                        max_scale = t.transpose(t.max(sample, 1).values.unsqueeze(0), 1, 0)                                                                                                                                     
                        sample = t.div(sample, max_scale)
                    pred = model(sample, sample_mask, 1)
                    if maxscale:
                        pred = t.mul(pred, max_scale)
                    x_np = np.concatenate((x_np, pred.cpu().detach().numpy()), axis=1)
                    x_mask_np = np.concatenate((x_mask_np, np.ones((x_np.shape[0], 1))), axis=1)
                forecasts.extend(x_np[:, -horizon:])
                logger.debug(f"forecasts[-1].shape: {forecasts[-1].shape}")
                logger.debug(f"x_np[:, -horizon:].shape = {x_np[:, -horizon:].shape}")
                logger.debug(f"x_np.shape = {x_np.shape}")             
        else:
            logger.info(f"Training model for {seasonal_pattern} pattern.")
            history_size_in_horizons = history_size[seasonal_pattern]
            horizon = M4Meta.horizons_map[seasonal_pattern]
            input_size = lookback * horizon
             
            # Training Set
            training_values = group_values(dataset.values, dataset.groups, seasonal_pattern)
                 
            dummy_forecast = np.zeros((len(training_values), horizon))
            forecasts.extend(dummy_forecast)


    forecasts_df = pd.DataFrame(forecasts, columns=[f'V{i + 1}' for i in range(np.max(M4Meta.horizons))])
    forecasts_df.index = dataset.ids
    forecasts_df.index.name = 'id'
    forecasts_df.to_csv('forecast.csv')

    training_set = M4Dataset.load(training=True)
    test_set = M4Dataset.load(training=False)

    forecast = np.array([v[~np.isnan(v)] for v in forecasts], dtype=object)

    grouped_smapes = {group_name:
                            np.mean(smape_2(forecast=group_values(values=forecast,
                                                                groups=test_set.groups,
                                                                group_name=group_name),
                                            target=group_values(values=test_set.values,
                                                                groups=test_set.groups,
                                                                group_name=group_name)))
                        for group_name in np.unique(test_set.groups)}
    grouped_smapes = summarize_groups(grouped_smapes, test_set)

    model_mases = {}
    for group_name in np.unique(test_set.groups):
        model_forecast = group_values(forecast, test_set.groups, group_name)
    
        target = group_values(test_set.values, test_set.groups, group_name)
        # all timeseries within group have same frequency
        frequency = training_set.frequencies[test_set.groups == group_name][0]
        insample = group_values(training_set.values, test_set.groups, group_name)

        model_mases[group_name] = np.mean([mase(forecast=model_forecast[i],
                                                insample=insample[i],
                                                outsample=target[i],
                                                frequency=frequency) for i in range(len(model_forecast))])
    grouped_model_mases = summarize_groups(model_mases, test_set)
    def round_all(d):
        return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))
    print(round_all(grouped_smapes))
    print(round_all(grouped_model_mases))
    model = model.double()
    preds = predict_M4(model=model, scale=True)
    res = score_M4(preds)
    print(res)


if __name__ == '__main__':
    print("test")
    main()