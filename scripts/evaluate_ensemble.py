import pandas as pd
import os
import glob
import click
from box import Box
import yaml
import sys

sys.path.append("")

from GPTime.utils.scoring import score_M4


@click.command()
@click.option("--cfg_path", required=True)
def evaluate_ensemble(cfg_path):
    with open(cfg_path, "r") as ymlfile:
        ensemble_cfg = Box(yaml.safe_load(ymlfile))
    
    experiment_folder = os.path.join(ensemble_cfg.storage_folder, ensemble_cfg.ensemble_name)
    print(experiment_folder)
    ensemble_members = glob.glob(os.path.join(*[experiment_folder, "**", "forecast.csv"]))
    ensemble_members = [memb for memb in ensemble_members if "seasonal" not in memb]
    print(ensemble_members)
    df_list = []
    for member_fname in ensemble_members:
        print(member_fname)
        df = pd.read_csv(member_fname)
        df_list.append(df)
    df_concat = pd.concat(df_list)
    by_row = df_concat.groupby(df_concat.index)
    df_median = by_row.median()
    df_mean = by_row.mean()
    preds_median = df_median.values
    preds_mean = df_mean.values
    
    print("scoring median")
    print(f"val_set: {ensemble_cfg.val_set}")
    res_median = score_M4(predictions=preds_median, df_results_name=os.path.join(experiment_folder, "result_median.csv"), val=ensemble_cfg.val_set)
    print("scoring mean")
    res_mean = score_M4(predictions=preds_mean, df_results_name=os.path.join(experiment_folder, "result_mean.csv"), val=ensemble_cfg.val_set)

    #df_median.to_csv(os.path.join(experiment_folder, "forecast_median.csv"))
    #df_mean.to_csv(os.path.join(experiment_folder, "forecast_mean.csv"))

    print(res_median)
    print(res_mean)

if __name__ == "__main__":
    evaluate_ensemble()
