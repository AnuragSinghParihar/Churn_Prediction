"""
test_models.py — Unit tests for model loading, prediction, and accuracy.

Tests validate:
  - All artefacts load without error
  - predict() output shapes and value ranges
  - Risk labels are in {Low, Medium, High}
  - Both models exceed minimum accuracy threshold on a held-out test set
"""

import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from src.inference import load_models, predict
from src.preprocessing import preprocess

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent / "data" / "online_gaming_behavior_dataset.csv"

# Minimum acceptable accuracy on the 20% hold-out set
MIN_ACCURACY = 0.80

VALID_RISK_LABELS = {"Low", "Medium", "High"}
VALID_PREDICTION_LABELS = {"Churned", "Retained"}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loaded_models():
    """Load all model artefacts once per test module."""
    return load_models()


@pytest.fixture(scope="module")
def sample_players():
    """Four clean sample players covering diverse categorical values."""
    return pd.DataFrame(
        {
            "Age": [25, 30, 22, 45],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Location": ["USA", "Europe", "Asia", "Other"],
            "GameGenre": ["Action", "Strategy", "RPG", "Simulation"],
            "PlayTimeHours": [5.5, 12.0, 2.0, 8.0],
            "InGamePurchases": [1, 0, 0, 1],
            "GameDifficulty": ["Medium", "Hard", "Easy", "Medium"],
            "SessionsPerWeek": [3, 10, 1, 5],
            "AvgSessionDurationMinutes": [45.0, 90.0, 20.0, 60.0],
            "PlayerLevel": [20, 50, 5, 35],
            "AchievementsUnlocked": [5, 25, 2, 15],
        }
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def test_all_artefacts_load(loaded_models):
    log_model, dt_model, scaler, label_encoders, feature_names = loaded_models
    assert log_model is not None
    assert dt_model is not None
    assert scaler is not None
    assert isinstance(label_encoders, dict)
    assert len(feature_names) == 11


def test_label_encoders_contain_expected_columns(loaded_models):
    _, _, _, label_encoders, _ = loaded_models
    for col in ["Gender", "Location", "GameGenre", "GameDifficulty"]:
        assert col in label_encoders, f"Missing encoder for column: {col}"


# ── Logistic Regression predictions ──────────────────────────────────────────

def test_lr_predict_output_lengths(loaded_models, sample_players):
    log_model, _, scaler, les, _ = loaded_models
    X_scaled, _ = preprocess(sample_players, les, scaler)
    results = predict(log_model, X_scaled)
    n = len(sample_players)
    assert len(results["prediction"]) == n
    assert len(results["churn_prob"]) == n
    assert len(results["risk"]) == n


def test_lr_churn_probabilities_in_unit_interval(loaded_models, sample_players):
    log_model, _, scaler, les, _ = loaded_models
    X_scaled, _ = preprocess(sample_players, les, scaler)
    results = predict(log_model, X_scaled)
    assert all(0.0 <= p <= 1.0 for p in results["churn_prob"]), \
        "Churn probabilities outside [0, 1]"


def test_lr_risk_labels_are_valid(loaded_models, sample_players):
    log_model, _, scaler, les, _ = loaded_models
    X_scaled, _ = preprocess(sample_players, les, scaler)
    results = predict(log_model, X_scaled)
    assert all(r in VALID_RISK_LABELS for r in results["risk"])


def test_lr_prediction_labels_are_valid(loaded_models, sample_players):
    log_model, _, scaler, les, _ = loaded_models
    X_scaled, _ = preprocess(sample_players, les, scaler)
    results = predict(log_model, X_scaled)
    assert all(p in VALID_PREDICTION_LABELS for p in results["prediction"])


# ── Decision Tree predictions ─────────────────────────────────────────────────

def test_dt_predict_output_lengths(loaded_models, sample_players):
    _, dt_model, scaler, les, _ = loaded_models
    _, X_encoded = preprocess(sample_players, les, scaler)
    results = predict(dt_model, X_encoded)
    assert len(results["churn_prob"]) == len(sample_players)


def test_dt_churn_probabilities_in_unit_interval(loaded_models, sample_players):
    _, dt_model, scaler, les, _ = loaded_models
    _, X_encoded = preprocess(sample_players, les, scaler)
    results = predict(dt_model, X_encoded)
    assert all(0.0 <= p <= 1.0 for p in results["churn_prob"])


def test_dt_risk_labels_are_valid(loaded_models, sample_players):
    _, dt_model, scaler, les, _ = loaded_models
    _, X_encoded = preprocess(sample_players, les, scaler)
    results = predict(dt_model, X_encoded)
    assert all(r in VALID_RISK_LABELS for r in results["risk"])


# ── Hold-out accuracy tests ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def holdout_data():
    """Build held-out test set using the same split parameters as train.py.

    Skips automatically when the dataset CSV is absent or empty (e.g. before
    the user runs `python -m src.download_data` to fetch the Kaggle dataset).
    """
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        pytest.skip(
            f"Dataset not found or empty at {DATA_PATH}. "
            "Run `python -m src.download_data` to fetch it."
        )
    df = pd.read_csv(DATA_PATH)
    les = joblib.load(MODELS_DIR / "label_encoders.pkl")
    scaler = joblib.load(MODELS_DIR / "scalers.pkl")

    # Binary target: Low engagement = 1 (Churn)
    y = (df["EngagementLevel"] == "Low").astype(int).values
    X_scaled, X_encoded = preprocess(df, les, scaler, validate=False)

    _, X_test_s, _, X_test_e, _, y_test = train_test_split(
        X_scaled, X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_test_s, X_test_e, y_test


def test_logistic_regression_accuracy_above_threshold(loaded_models, holdout_data):
    log_model = loaded_models[0]
    X_test_scaled, _, y_test = holdout_data
    acc = accuracy_score(y_test, log_model.predict(X_test_scaled))
    assert acc >= MIN_ACCURACY, (
        f"Logistic Regression accuracy {acc:.3f} is below threshold {MIN_ACCURACY}. "
        "Retrain with `python -m src.train`."
    )


def test_decision_tree_accuracy_above_threshold(loaded_models, holdout_data):
    dt_model = loaded_models[1]
    _, X_test_encoded, y_test = holdout_data
    acc = accuracy_score(y_test, dt_model.predict(X_test_encoded))
    assert acc >= MIN_ACCURACY, (
        f"Decision Tree accuracy {acc:.3f} is below threshold {MIN_ACCURACY}. "
        "Retrain with `python -m src.train`."
    )
