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
    step1_dfs = []
    step2_dfs = []
    init_dfs = []
    dirs = glob.glob(os.path.join(folder_path, "**/"))
    print(dirs)
    for d in dirs:
        print(d)
        fnames = [f for f in glob.glob(os.path.join(d, "*")) if "forecast" in f]
        dc = {}
        dc["step1"] = [f for f in fnames if "_step1" in f]
        dc["step2"] = [f for f in fnames if "_step2" in f]
        dc["init"]  = [f for f in fnames if "_init" in f]
        for name, l in dc.items():
            l.sort()
            dfs = []
            for fname in l:
                print(fname)
                freq = fname.split("/")[-1][0]
                print(f"freq: {freq}")
                df = pd.read_csv(fname, index_col=0)
                df = df[df.index.str.contains(freq, na=False)]
                dfs.append(df)
            df = pd.concat(dfs, sort=False)
            result_fname = os.path.join(d, f"{name}_forecast.csv")
            df.to_csv(result_fname)
            if name == "step1":
                step1_dfs.append(df)
            elif name == "step2":
                step2_dfs.append(df)
            elif name == "init":
                init_dfs.append(df)
            else:
                print("WARNING: Couldnæt find name of df")
            res = score_M4(predictions=df.values, df_results_name=os.path.join(d, f"{name}_result_median.csv"), val=False)
            print(res)
    step1_fnames = glob.glob(os.path.join(folder_path, "**/step1_forecast.csv"))
    step2_fnames = glob.glob(os.path.join(folder_path, "**/step2_forecast.csv"))
    init_fnames = glob.glob(os.path.join(folder_path, "**/init_forecast.csv"))
    print(step1_fnames)
    print(step2_fnames)
    print(init_fnames)
    
    csv_dict = {"step1": step1_fnames, "step2": step2_fnames, "init": init_fnames}
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


if __name__ == "__main__":
    evaluate_ensemble()
