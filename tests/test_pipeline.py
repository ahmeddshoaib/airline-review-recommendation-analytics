from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.airline_analytics import RATING_COLUMNS, TrainingOnlyAirlineMedianImputer, prepare_reviews


class AirlinePipelineTests(unittest.TestCase):
    def test_unseen_airline_uses_training_global_median(self) -> None:
        training = pd.DataFrame(
            {
                "Airline_Name": ["A", "A", "B", "B"],
                **{column: [1.0, 3.0, 4.0, 5.0] for column in RATING_COLUMNS},
            }
        )
        test = pd.DataFrame(
            {"Airline_Name": ["Unseen"], **{column: [np.nan] for column in RATING_COLUMNS}}
        )
        imputer = TrainingOnlyAirlineMedianImputer().fit(training)
        transformed = imputer.transform(test)
        self.assertTrue((transformed[RATING_COLUMNS].iloc[0] == 3.5).all())

    def test_target_cleaning(self) -> None:
        frame = pd.DataFrame(
            {
                "Airline_Name": ["A", "B"],
                "Review": ["good", "bad"],
                "Recommended": ["Yes", " no "],
                **{column: [4, 2] for column in RATING_COLUMNS},
            }
        )
        prepared = prepare_reviews(frame)
        self.assertEqual(prepared["target"].tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()

