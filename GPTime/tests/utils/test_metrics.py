import unittest
import pandas as pd
import numpy as np
import logging
import sys
sys.path.append("")

logger = logging.getLogger(__name__)

from GPTime.config import cfg

from GPTime.utils.metrics import SMAPE, MASE, OWA

## Pre load and clean data

# test data
df_test_h = pd.read_csv(cfg.tests.data.M4.Htest, index_col=0)
df_test_d = pd.read_csv(cfg.tests.data.M4.Dtest, index_col=0)
df_test_w = pd.read_csv(cfg.tests.data.M4.Wtest, index_col=0)
df_test_m = pd.read_csv(cfg.tests.data.M4.Mtest, index_col=0)
df_test_q = pd.read_csv(cfg.tests.data.M4.Qtest, index_col=0)
df_test_y = pd.read_csv(cfg.tests.data.M4.Ytest, index_col=0)
# train data for scaling
df_train_h = pd.read_csv(cfg.tests.data.M4.Htrain, index_col=0)
df_train_d = pd.read_csv(cfg.tests.data.M4.Dtrain, index_col=0)
df_train_w = pd.read_csv(cfg.tests.data.M4.Wtrain, index_col=0)
df_train_m = pd.read_csv(cfg.tests.data.M4.Mtrain, index_col=0)
df_train_q = pd.read_csv(cfg.tests.data.M4.Qtrain, index_col=0)
df_train_y = pd.read_csv(cfg.tests.data.M4.Ytrain, index_col=0)
# submission tests
df_smyl = pd.read_csv(cfg.tests.data.M4.smyl, index_col=0)
df_montero = pd.read_csv(cfg.tests.data.M4.montero, index_col=0)
df_naive = pd.read_csv(cfg.tests.data.M4.naive, index_col=0)
df_naive2 = pd.read_csv(cfg.tests.data.M4.naive2, index_col=0)
# create full test df
frames = [df_test_y, df_test_q, df_test_m, df_test_w, df_test_d, df_test_h]
df_test = pd.concat(frames)
# train df for scaling in mase
frames = [
    (df_train_y, cfg.scoring.m4.periods.yearly),
    (df_train_q, cfg.scoring.m4.periods.quarterly),
    (df_train_m, cfg.scoring.m4.periods.monthly),
    (df_train_w, cfg.scoring.m4.periods.weekly), 
    (df_train_d, cfg.scoring.m4.periods.daily), 
    (df_train_h, cfg.scoring.m4.periods.hourly), 
]

scale_list = []
for df, period in frames:
    s = (df.diff(periods=period, axis=1).abs().mean(axis=1).reset_index(drop=True))
    scale_list.append(s.values)
scale = np.concatenate(scale_list)

class TestMetrics(unittest.TestCase):
    """
    Test submissions from M4.
    """
    def setUp(self):
        self.test_data = df_test.values
        self.smyl_predictions = df_smyl.values
        self.montero_predictions = df_montero.values
        self.naive_predictions = df_naive.values
        self.naive2_predictions = df_naive2.values
        self.scale = scale

    def test_smape(self):
        smape_smyl = round(SMAPE(self.test_data, self.smyl_predictions),3)
        smape_montero = round(SMAPE(self.test_data, self.montero_predictions),3)
        smape_naive = round(SMAPE(self.test_data, self.naive_predictions),3)
        smape_naive2 = round(SMAPE(self.test_data, self.naive2_predictions),3)
        self.assertEqual(smape_smyl, 11.374, "Should be 11.374")
        self.assertEqual(montero_smyl, 11.720, "Should be 11.720")
        self.assertEqual(smape_naive, 14.208, "Should be 14.208")
        self.assertEqual(smape_naive2, 13.564, "Should be 13.564")

    def test_mase(self):
        mase_smyl = round(MASE(self.test_data, self.smyl_predictions, scale),3)
        mase_montero = round(MASE(self.test_data, self.montero_predictions, scale),3)
        mase_naive = round(MASE(self.test_data, self.naive_predictions, scale),3)
        mase_naive2 = round(MASE(self.test_data, self.naive2_predictions, scale),3)
        self.assertEqual(mase_smyl, 1.536, "Should be 1.536")
        self.assertEqual(mase_montero, 1.551, "Should be 1.551")
        self.assertEqual(mase_naive, 2.044, "Should be 2.044")
        self.assertEqual(mase_naive2, 1.912, "Should be 1.912")

    def test_owa(self):
        smape_smyl = SMAPE(self.test_data, self.smyl_predictions)
        smape_montero = SMAPE(self.test_data, self.montero_predictions)
        smape_naive = SMAPE(self.test_data, self.naive_predictions)
        smape_naive2 = SMAPE(self.test_data, self.naive2_predictions)
        mase_smyl = MASE(self.test_data, self.smyl_predictions, scale)
        mase_montero = MASE(self.test_data, self.montero_predictions, scale)
        mase_naive = MASE(self.test_data, self.naive_predictions, scale)
        mase_naive2 = MASE(self.test_data, self.naive2_predictions, scale)
        owa_smyl = round(OWA(mase_smyl, smape_smyl),3)
        owa_montero = round(OWA(mase_montero, smape_montero),3)
        owa_naive = round(OWA(mase_naive, smape_naive),3)
        owa_naive2 = round(OWA(mase_naive2, smape_naive2),3)
        self.assertEqual(owa_smyl, 0.821, "Should be 0.821")
        self.assertEqual(owa_montero, 0.838, "Should be 0.838")
        self.assertEqual(owa_naive, 1.058, "Should be 1.058")
        self.assertEqual(owa_naive2, 1.000, "Should be 1.000")

if __name__ == '__main__':
    unittest.main()