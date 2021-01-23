import glob
import os
import pandas as pd


def fix_forecasts():
    folder_path = "storage/experiments/almost_global_test4/"
    forecast_fnames = glob.glob(folder_path + "**/forecast.csv")
    freq_order = {"Y": 1, "Q": 2, "M": 3, "W": 4, "D": 5, "H": 6}
    for fname in forecast_fnames:
        print(fname)
        df = pd.read_csv(fname)
        df["freq"] = df["id"].astype(str).str[0]
        df["in_freq_rank"] = df["id"].astype(str).str[1:].astype(int)
        df["rank"] = df["freq"].apply(lambda x: freq_order.get(x))
        df = df.sort_values(["rank", "in_freq_rank"])
        df = df.drop(columns=["freq", "rank", "in_freq_rank"])
        new_path = "/".join(fname.split("/")[:-1]) + "/nbeats_forecast.csv"
        df.to_csv(new_path, index=False)
    print("Done.")
if __name__ == "__main__":
    fix_forecasts()
