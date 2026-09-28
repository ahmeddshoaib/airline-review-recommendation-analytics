from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


class OutputTests(unittest.TestCase):
    def test_metrics_are_labelled_synthetic(self) -> None:
        metrics = pd.read_csv(Path("outputs/model_metrics.csv"))
        self.assertTrue(metrics["data_label"].eq("synthetic_demo").all())
        self.assertEqual(len(metrics), 5)
        self.assertTrue(metrics[["f1", "roc_auc"]].notna().all().all())


if __name__ == "__main__":
    unittest.main()

