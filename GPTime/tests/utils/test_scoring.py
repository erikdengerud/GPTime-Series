# either random mlp or naive or both?
import unittest
import sys
import pandas as pd
import numpy as np

sys.path.append("")
from GPTime.utils.scoring import predict_M4, score_M4

# Dummy model:
import torch.nn as nn
import torch.nn.functional as F


class Naive(nn.Module):
    def __init__(self):
        super(Naive, self).__init__()
        self.memory = 1

    def forward(self, x):
        return x


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


class TestMetrics(unittest.TestCase):
    """
    Test that the predict and scoring function works.
    """

    def setUp(self):
        self.naive = Naive().double()
        self.mlp = MLP(in_features=10, out_features=1, n_hidden=16).double()

    def test_predictM4(self):
        naive_predictions = predict_M4(model=self.naive)
        mlp_predictions = predict_M4(model=self.mlp)

        # numpy array
        self.assertIsInstance(naive_predictions, np.ndarray, "Should be np.array")
        self.assertIsInstance(mlp_predictions, np.ndarray, "Should be np.array")
        # shape
        self.assertEqual(
            naive_predictions.shape, (100000, 48), "Should be (100000, 48)"
        )
        self.assertEqual(mlp_predictions.shape, (100000, 48), "Should be (100000, 48)")

    def test_scoreM4(self):
        rand_int = np.random.randint(999)
        naive_predictions = predict_M4(model=self.naive)
        mlp_predictions = predict_M4(model=self.mlp)
        naive_scores = score_M4(
            naive_predictions, f"GPTime/tests/results/M4/naive_test{rand_int}.csv"
        )
        mlp_scores = score_M4(
            mlp_predictions, f"GPTime/tests/results/M4/mlp_test{rand_int}.csv"
        )

        # scores were saved
        df_naive = pd.read_csv(
            f"GPTime/tests/results/M4/naive_test{rand_int}.csv", index_col=0
        )
        df_mlp = pd.read_csv(
            f"GPTime/tests/results/M4/mlp_test{rand_int}.csv", index_col=0
        )

        self.assertFalse(df_naive.empty, msg="Should be False")
        self.assertFalse(df_mlp.empty, msg="Should be False")

        # scores have right format
        self.assertEqual(df_naive.shape, (7, 3), "Shape should be (7,3)")
        self.assertEqual(df_mlp.shape, (7, 3), "Shape should be (7,3)")

        # smape 0-200
        # mase > 0
        # owa > 0


if __name__ == "__main__":
    unittest.main(verbosity=2)