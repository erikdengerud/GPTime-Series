"""
Data preprocessing.
"""
from typing import Dict, Tuple, Sequence
import glob
import pandas as pd
import numpy as np
import sys

sys.path.append("")


def preprocess_dataset(
    experiments: Dict,
    config: Dict,
    periods: Dict,
    all_train_files: Sequence,
    all_test_files: Sequence,
) -> Dict:
    datasets = {}

    for experiment in experiments.keys():
        if experiments[experiment]:
            print(f"Creating datasets for experiment {experiment}.")
            if experiment == "GLOBAL":
                train_files = all_train_files
                test_files = all_test_files
            else:
                train_files = []
                test_files = []
                for f in all_train_files:
                    if experiment.lower() in f.lower():
                        train_files.append(f)
                for f in all_test_files:
                    if experiment.lower() in f.lower():
                        test_files.append(f)

            # Finding longest possible lag value for the dataset
            num_non_na_train = []
            num_non_na_test = []
            for train_f, test_f in zip(train_files, test_files):
                df_train = pd.read_csv(train_f, index_col=0)
                df_test = pd.read_csv(test_f, index_col=0)
                num_non_na_train.append(min(df_train.count(axis=1).values))
                num_non_na_test.append(min(df_test.count(axis=1).values))
                print(train_f, num_non_na_train, num_non_na_test)
            num_non_na_train_int = min(num_non_na_train)
            num_non_na_test_int = min(num_non_na_test)
            print(f"num_non_na_train_int = {num_non_na_train_int}")
            print(f"num_non_na_test_int = {num_non_na_test_int}")

            # Creating the dataset
            l_train = []
            l_test = []
            for train_f, test_f in zip(train_files, test_files):
                l_train_tmp = []
                l_test_tmp = []
                df_train = pd.read_csv(train_f, index_col=0)
                df_test = pd.read_csv(test_f, index_col=0)
                Y_train = df_train.to_numpy()
                Y_test = df_test.to_numpy()
                assert (
                    Y_train.shape[0] == Y_test.shape[0]
                ), f"Y_train.shape[0], {Y_train.shape[0]} differs from Y_test.shape[0], {Y_test.shape[0]}"
                for i in range(Y_train.shape[0]):
                    s_train = Y_train[i][~np.isnan(Y_train[i])]
                    s_test = Y_test[i][~np.isnan(Y_test[i])]
                    l_train_tmp.append(
                        s_train[-num_non_na_train_int:]
                    )  # shortest in the train set
                    l_test_tmp.append(
                        s_test[:num_non_na_test_int]
                    )  # shortest in the test set

                # Scaling using the scaling in MASE
                if config["MASE_SCALE"]:
                    if config["MASE_SCALE_SEASONALITY"]:
                        df_train_out = pd.DataFrame(l_train_tmp)
                        df_test_out = pd.DataFrame(l_test_tmp)
                        # Find the seasonality of the ts
                        for p in periods.keys():
                            if p.lower() in train_f.lower():
                                s = periods[p]
                        if config["SCALE_SET"] == "FULL":
                            scale = (
                                df_train.diff(periods=s, axis=1)
                                .abs()
                                .mean(axis=1)
                                .reset_index(drop=True)
                            )
                        elif config["SCALE_SET"] == "SUB":
                            scale = (
                                df_train_out.diff(periods=s, axis=1)
                                .abs()
                                .mean(axis=1)
                                .reset_index(drop=True)
                            )
                        l_train_tmp = df_train_out.div(scale, axis=0).values.tolist()
                        l_test_tmp = df_test_out.div(scale, axis=0).values.tolist()
                    else:
                        df_train_out = pd.DataFrame(l_train_tmp)
                        df_test_out = pd.DataFrame(l_test_tmp)
                        if config["SCALE_SET"] == "FULL":
                            scale = (
                                df_train.diff(axis=1)
                                .abs()
                                .mean(axis=1)
                                .reset_index(drop=True)
                            )
                        elif config["SCALE_SET"] == "SUB":
                            scale = (
                                df_train_out.diff(axis=1)
                                .abs()
                                .mean(axis=1)
                                .reset_index(drop=True)
                            )
                        l_train_tmp = df_train_out.div(scale, axis=0).values.tolist()
                        l_test_tmp = df_test_out.div(scale, axis=0).values.tolist()

                for i in range(len(l_train_tmp)):
                    l_train.append(l_train_tmp[i])
                    l_test.append(l_test_tmp[i])

            X_train = np.array(l_train)
            X_test = np.array(l_test)
            print(f"X_train.shape = {X_train.shape}")
            print(f"X_test.shape = {X_test.shape}")
            datasets[experiment] = (X_train, X_test)

    # Check for Inf values that can occur during scaling
    for d in datasets.keys():
        if (
            len(np.argwhere(np.isinf(datasets[d][0])))
            + len(np.argwhere(np.isinf(datasets[d][1])))
            != 0
        ):
            print(f"WARNING: Inf values in dataset {d}.")
            train_inf = list(set(np.argwhere(np.isinf(datasets[d][0]))[:, 0]))
            X_train = np.delete(datasets[d][0], train_inf, 0)
            X_test = np.delete(datasets[d][1], train_inf, 0)
            test_inf = list(set(np.argwhere(np.isinf(X_test))[:, 0]))
            X_test = np.delete(X_test, test_inf, 0)
            X_train = np.delete(X_train, test_inf, 0)
            datasets[d] = (X_train, X_test)

    print("\n")
    print(f"Done. Created datasets for {[d for d in datasets.keys()]}.\n")
    print("Sizes of the datasets: ")
    for dataset in datasets.keys():
        print(
            f"{dataset:9} : ({str(datasets[dataset][0].shape[0]):>6}, {str(datasets[dataset][0].shape[1]):>3}), ({datasets[dataset][1].shape[0]:>6}, {datasets[dataset][1].shape[1]:>2})"
        )

    print("\n")
    for d in datasets.keys():
        if (
            len(np.argwhere(np.isinf(datasets[d][0])))
            + len(np.argwhere(np.isinf(datasets[d][1])))
            != 0
        ):
            print("Unsuccessful in removing Inf values.")
            print(f"WARNING: Inf values in dataset {d}.")

    if experiments["GLOBAL"]:
        X_train, X_test = datasets["GLOBAL"]
        # create dfs and write

    global_train = datasets["GLOBAL"][0]
    np.save("M4_GLOBAL_train.npy", global_train)
    global_test = datasets["GLOBAL"][1]
    np.save("M4_GLOBAL_test.npy", global_test)
    return datasets


if __name__ == "__main__":
    experiments = {
        "GLOBAL": True,
        "YEARLY": False,
        "QUARTERLY": False,
        "MONTHLY": False,
        "WEEKLY": False,
        "DAILY": False,
        "HOURLY": False,
    }

    config = {
        "MASE_SCALE": True,
        "MASE_SCALE_SEASONALITY": True,
        "SCALE_SET": "FULL",  # FULL or SUB. Either the all the data or just the last 12 obs.
    }

    # Periods used in MASE scaling
    # M4 paper defines yearly, weekly and daily to be 1

    periods = {
        "YEARLY": 1,
        "QUARTERLY": 4,
        "MONTHLY": 12,
        "WEEKLY": 1,
        "DAILY": 1,
        "HOURLY": 24,
    }
    all_test_files = glob.glob("GPTime/data/raw/M4/M4test/*")
    all_train_files = glob.glob("GPTime/data/raw/M4/M4train/*")
    all_train_files.sort()
    all_test_files.sort()
    print(all_train_files)
    print(all_test_files)
    """
    datasests = preprocess_dataset(
        experiments,
        config,
        periods,
        all_train_files,
        all_test_files,
    )
    """
