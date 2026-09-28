"""Leakage-safe customer recommendation modelling for airline reviews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RATING_COLUMNS = [
    "Overall_Rating",
    "Seat_Comfort",
    "Cabin_Staff_Service",
    "Food_and_Beverages",
    "Ground_Service",
    "Inflight_Entertainment",
    "Wifi_and_Connectivity",
    "Value_For_Money",
]
REQUIRED_COLUMNS = {"Airline_Name", "Review", "Recommended", *RATING_COLUMNS}


def clean_column_names(columns: pd.Index) -> pd.Index:
    return columns.str.strip().str.replace(" ", "_", regex=False).str.replace("&", "and", regex=False)


def load_reviews(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Input must be CSV or Excel")
    frame.columns = clean_column_names(frame.columns)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame


def prepare_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Recommended"] = data["Recommended"].astype(str).str.strip().str.lower()
    data["target"] = data["Recommended"].map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if data["target"].isna().any():
        raise ValueError("Recommended must contain only yes/no or 1/0")
    data["Airline_Name"] = data["Airline_Name"].fillna("Unknown").astype(str).str.strip()
    data["Review"] = data["Review"].fillna("").astype(str)
    for column in RATING_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        upper = 10 if column == "Overall_Rating" else 5
        data.loc[~data[column].between(0, upper), column] = np.nan
    return data


@dataclass
class TrainingOnlyAirlineMedianImputer:
    rating_columns: tuple[str, ...] = tuple(RATING_COLUMNS)

    def fit(self, frame: pd.DataFrame) -> "TrainingOnlyAirlineMedianImputer":
        self.airline_medians_ = frame.groupby("Airline_Name")[list(self.rating_columns)].median()
        self.global_medians_ = frame[list(self.rating_columns)].median()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "global_medians_"):
            raise RuntimeError("Imputer must be fitted on training data first")
        result = frame.copy()
        for column in self.rating_columns:
            airline_fill = result["Airline_Name"].map(self.airline_medians_[column])
            result[column] = result[column].fillna(airline_fill).fillna(self.global_medians_[column])
        return result


def metric_row(name: str, y_true: pd.Series, probabilities: np.ndarray, data_label: str) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "data_label": data_label,
        "model": name,
        "n_test": len(y_true),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def evaluate_models(data: pd.DataFrame, data_label: str, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        data,
        test_size=0.25,
        stratify=data["target"],
        random_state=random_state,
    )
    imputer = TrainingOnlyAirlineMedianImputer().fit(train)
    train_imp = imputer.transform(train)
    test_imp = imputer.transform(test)

    scaler = StandardScaler()
    x_train_num = scaler.fit_transform(train_imp[RATING_COLUMNS])
    x_test_num = scaler.transform(test_imp[RATING_COLUMNS])
    y_train, y_test = train["target"], test["target"]

    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    logistic.fit(x_train_num, y_train)
    logistic_probability = logistic.predict_proba(x_test_num)[:, 1]

    forest = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    forest.fit(train_imp[RATING_COLUMNS], y_train)
    forest_probability = forest.predict_proba(test_imp[RATING_COLUMNS])[:, 1]

    vectorizer = TfidfVectorizer(min_df=3, max_features=6000, ngram_range=(1, 2), stop_words="english")
    x_train_text = vectorizer.fit_transform(train["Review"])
    x_test_text = vectorizer.transform(test["Review"])
    text_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    text_model.fit(x_train_text, y_train)
    text_probability = text_model.predict_proba(x_test_text)[:, 1]

    combined_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    combined_model.fit(hstack([x_train_num, x_train_text]), y_train)
    combined_probability = combined_model.predict_proba(hstack([x_test_num, x_test_text]))[:, 1]

    metrics = pd.DataFrame(
        [
            metric_row("numerical_logistic", y_test, logistic_probability, data_label),
            metric_row("numerical_random_forest", y_test, forest_probability, data_label),
            metric_row("text_logistic", y_test, text_probability, data_label),
            metric_row("combined_logistic", y_test, combined_probability, data_label),
        ]
    )
    importance = pd.DataFrame(
        {
            "feature": RATING_COLUMNS,
            "random_forest_importance": forest.feature_importances_,
            "logistic_coefficient": logistic.coef_[0],
        }
    ).sort_values("random_forest_importance", ascending=False)
    return metrics, importance


def evaluate_airline_holdout(data: pd.DataFrame, data_label: str, random_state: int = 42) -> pd.DataFrame:
    counts = data["Airline_Name"].value_counts()
    eligible = counts[counts >= max(20, int(0.005 * len(data)))].index.to_numpy()
    if len(eligible) < 5:
        raise ValueError("At least five sufficiently represented airlines are required for holdout evaluation")
    rng = np.random.default_rng(random_state)
    test_airlines = set(rng.choice(eligible, size=max(1, len(eligible) // 5), replace=False))
    train = data[~data["Airline_Name"].isin(test_airlines)].copy()
    test = data[data["Airline_Name"].isin(test_airlines)].copy()

    imputer = TrainingOnlyAirlineMedianImputer().fit(train)
    train_imp, test_imp = imputer.transform(train), imputer.transform(test)
    model = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(train_imp[RATING_COLUMNS], train["target"])
    probabilities = model.predict_proba(test_imp[RATING_COLUMNS])[:, 1]
    row = metric_row("airline_holdout_random_forest", test["target"], probabilities, data_label)
    row["n_train_airlines"] = train["Airline_Name"].nunique()
    row["n_test_airlines"] = test["Airline_Name"].nunique()
    return pd.DataFrame([row])


def save_figures(metrics: pd.DataFrame, importance: pd.DataFrame, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    chart = metrics.set_index("model")[["f1", "roc_auc"]].sort_values("roc_auc")
    ax = chart.plot(kind="barh", figsize=(10, 5), color=["#2563eb", "#7c3aed"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    ax.set_title("Synthetic validation only: model comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(directory / "model_comparison.png", dpi=180)
    plt.close()

    ordered = importance.sort_values("random_forest_importance")
    ax = ordered.plot.barh(
        x="feature",
        y="random_forest_importance",
        figsize=(10, 5),
        legend=False,
        color="#0f766e",
    )
    ax.set_xlabel("Random-forest importance")
    ax.set_ylabel("")
    ax.set_title("Synthetic validation only: numerical feature importance")
    plt.tight_layout()
    plt.savefig(directory / "numeric_feature_importance.png", dpi=180)
    plt.close()

