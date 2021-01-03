"""
Create a train and test set out of the M4 train dataset.
"""
import pandas as pd
import numpy
import os
import glob
import numpy as np

def create_validation():
    horizons = {
        "Yearly" : 6,
        "Quarterly" : 8,
        "Monthly" : 18,
        "Weekly" : 13,
        "Daily" : 14,
        "Hourly" : 48,
    }
    m4_train_path = "GPTime/data/raw/M4/M4train"
    m4_val_train_path = "GPTime/data/processed/M4_val_train"
    m4_val_test_path = "GPTime/data/processed/M4_val_test"
    m4_train_files = glob.glob(os.path.join(m4_train_path, "*"))

    print(m4_train_files)
    for fname in m4_train_files:
        for h in horizons.keys():
            if h.lower() in fname.lower():
                horizon = horizons[h]
                str_h = h
        print(f"Processing {fname}")
        Y = pd.read_csv(fname, index_col=0).to_numpy()
        new_train = []
        new_test = []
        new_Y_train = np.empty(Y.shape)
        new_Y_test = np.empty((Y.shape[0], horizon))
        new_Y_train[:] = np.nan
        new_Y_test[:] = np.nan

        print(horizon)
        for i in range(Y.shape[0]):
            ts = Y[i][~np.isnan(Y[i])]
            train = ts[:-horizon]
            test = ts[-horizon:]
            new_Y_train[i][0:len(train)] = train
            new_Y_test[i][0:len(test)] = test
        train_index = [str_h[0]+str(i) for i in range(new_Y_train.shape[0])]
        test_index = [str_h[0]+str(i) for i in range(new_Y_test.shape[0])]
        df_train = pd.DataFrame(data=new_Y_train, index=train_index)
        df_test = pd.DataFrame(data=new_Y_test, index=test_index)
        df_train.columns = ["V" + str(i) for i in range(new_Y_train.shape[1])]
        df_test.columns = ["V" + str(i) for i in range(new_Y_test.shape[1])]
        
        tmp_name = fname.split("/")[-1]
        print(tmp_name)
        df_train.to_csv(os.path.join(m4_val_train_path, tmp_name))
        tmp_name = tmp_name.split("-")[0] + "-test.csv"
        print(tmp_name)
        df_test.to_csv(os.path.join(m4_val_test_path, tmp_name))
        print(df_train.head())
        print(df_test.head())

if __name__ == "__main__":
    create_validation()