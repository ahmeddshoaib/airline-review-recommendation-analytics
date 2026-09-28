"""Run the airline recommendation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.airline_analytics import (  # noqa: E402
    evaluate_airline_holdout,
    evaluate_models,
    load_reviews,
    prepare_reviews,
    save_figures,
)


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--data-label", required=True, choices=["synthetic_demo", "authorised_source"])
args = parser.parse_args()

data = prepare_reviews(load_reviews(args.data))
metrics, importance = evaluate_models(data, args.data_label)
holdout = evaluate_airline_holdout(data, args.data_label)
metrics = pd.concat([metrics, holdout], ignore_index=True, sort=False)

output = ROOT / "outputs"
output.mkdir(exist_ok=True)
metrics.to_csv(output / "model_metrics.csv", index=False)
importance.to_csv(output / "numeric_feature_importance.csv", index=False)
save_figures(metrics, importance, ROOT / "figures")

print(metrics.to_string(index=False))
print("ANALYSIS COMPLETED")

