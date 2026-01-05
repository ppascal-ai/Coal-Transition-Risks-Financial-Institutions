from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from data_loader import get_clean_data

DEFAULT_LIFETIME_YEARS = 35
FORECAST_END_YEAR = 2030


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    feature_names: List[str]
    accuracy: float
    roc_auc: float | None
    feature_importances: pd.DataFrame | None


def _prepare_training_data(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature/target sets from retired units with sufficient data.
    Target: retired before completing DEFAULT_LIFETIME_YEARS.
    """
    retired = clean_df.dropna(subset=["Retired year", "Start year"]).copy()
    retired["Start year"] = pd.to_numeric(retired["Start year"], errors="coerce")
    retired["Retired year"] = pd.to_numeric(retired["Retired year"], errors="coerce")
    retired = retired.dropna(subset=["Start year", "Retired year"])

    retired["operating_years"] = retired["Retired year"] - retired["Start year"]
    retired = retired[retired["operating_years"] >= 0]  # discard invalid records

    retired["early_closure"] = retired["operating_years"] < DEFAULT_LIFETIME_YEARS

    features = retired[
        [
            "Capacity (MW)",
            "Country/Area",
            "Coal type",
            "Combustion technology",
            "Subregion",
            "Region",
            "Start year",
        ]
    ]
    target = retired["early_closure"].astype(int)
    return features, target


def train_early_closure_model(clean_df: pd.DataFrame) -> ModelArtifacts:
    """
    Train a Random Forest model to predict early closures.
    """
    X, y = _prepare_training_data(clean_df)
    numeric_features = ["Capacity (MW)", "Start year"]
    categorical_features = ["Country/Area", "Coal type", "Combustion technology", "Subregion", "Region"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = None
    roc_auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    if y.nunique() > 1 and y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            roc_auc = None

    # Compute feature importances from the fitted forest on transformed features.
    feature_importances_df = None
    try:
        preproc = model.named_steps["preprocessor"]
        transformed_features = preproc.get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        feature_importances_df = (
            pd.DataFrame({"feature": transformed_features, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        feature_importances_df = None

    return ModelArtifacts(
        pipeline=model,
        feature_names=X.columns.tolist(),
        accuracy=accuracy,
        roc_auc=roc_auc,
        feature_importances=feature_importances_df,
    )


def score_current_fleet(clean_df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """
    Apply the trained model to non-retired units and compute risk of early closure.
    """
    current = clean_df[clean_df["Status"] != "retired"].copy()
    current["Start year"] = pd.to_numeric(current["Start year"], errors="coerce")
    current = current.dropna(subset=["Start year"])

    features = current[
        [
            "Capacity (MW)",
            "Country/Area",
            "Coal type",
            "Combustion technology",
            "Subregion",
            "Region",
            "Start year",
        ]
    ]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
    else:
        probs = model.predict(features)
    current["early_closure_risk"] = probs
    current["natural_retirement_year"] = current["Start year"] + DEFAULT_LIFETIME_YEARS

    # Flag plants that could close before 2030 (either by early closure or by natural retirement within horizon).
    current["at_risk_before_2030"] = (
        (current["natural_retirement_year"] <= FORECAST_END_YEAR)
        | (current["early_closure_risk"] >= 0.5)
    )
    return current


def top_at_risk_plants(current_scored: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Return the top-N plants with highest early closure risk where early closure matters before 2030.
    """
    candidates = current_scored[
        (current_scored["natural_retirement_year"] > FORECAST_END_YEAR)
        & (current_scored["at_risk_before_2030"])
    ]
    return candidates.sort_values("early_closure_risk", ascending=False).head(top_n)[
        ["Plant name", "Unit name", "Country/Area", "early_closure_risk", "natural_retirement_year"]
    ]


if __name__ == "__main__":
    clean_df = get_clean_data()
    artifacts = train_early_closure_model(clean_df)
    scored = score_current_fleet(clean_df, artifacts.pipeline)
    at_risk = top_at_risk_plants(scored, top_n=5)

    print(f"Accuracy: {artifacts.accuracy:.3f}")
    if artifacts.roc_auc is not None:
        print(f"ROC-AUC: {artifacts.roc_auc:.3f}")
    else:
        print("ROC-AUC: not available (single-class or insufficient variation).")
    print("Top 5 at-risk plants (probability of early closure):")
    print(at_risk.to_string(index=False))
