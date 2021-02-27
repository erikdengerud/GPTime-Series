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
@click.option("--folder_path", required=True)
def evaluate_ensemble(folder_path):
    
    csvs = [f for f in glob.glob(os.path.join(folder_path, "**/*csv")) if "forecast" in f]
    csvs = [f for f in csvs if "test_name" in f]
    step_1_csvs = [f for f in csvs if "step1" in f]
    step_2_csvs = [f for f in csvs if "step2" in f]
    init_csvs = [f for f in csvs if "initialization" in f]
    csv_dict = {"step1": step_1_csvs, "step2": step_2_csvs, "init": init_csvs}
    for name, csv_list in csv_dict.items():
        df_list = []
        for csv_fname in csv_list:
            print(csv_fname)
            df = pd.read_csv(csv_fname)
            df_list.append(df)
        df_concat = pd.concat(df_list)
        by_row = df_concat.groupby(df_concat.index)
        df_median = by_row.median()
        df_mean = by_row.mean()
        preds_median = df_median.values
        preds_mean = df_mean.values
        print("scoring median")
        res_median = score_M4(predictions=preds_median, df_results_name=os.path.join(folder_path, f"{name}_result_median.csv"), val=False)
        print("scoring mean")
        res_mean = score_M4(predictions=preds_mean, df_results_name=os.path.join(folder_path, f"{name}_result_mean.csv"), val=False)

        #df_median.to_csv(os.path.join(experiment_folder, "forecast_median.csv"))
        #df_mean.to_csv(os.path.join(experiment_folder, "forecast_mean.csv"))

        print(res_median)
        print(res_mean)
    """
    step_1_csvs = 
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
    """

if __name__ == "__main__":
    evaluate_ensemble()
