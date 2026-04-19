"""
inference.py — Model loading and prediction for Player Churn Prediction.

Models encode the target as a binary label:
  0 → Retained  (Medium or High engagement)
  1 → Churned   (Low engagement)

predict_proba()[:, 1] therefore gives P(Churn) for every row.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
MODELS_DIR = _ROOT / "models"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(models_dir: Path | str | None = None):
    """Load all serialised models and preprocessing artefacts.

    Parameters
    ----------
    models_dir : path-like, optional
        Override the default models/ directory.

    Returns
    -------
    tuple
        (log_model, dt_model, scaler, label_encoders, feature_names)

    Raises
    ------
    FileNotFoundError
        If any expected pickle file is absent.
    """
    d = Path(models_dir) if models_dir else MODELS_DIR
    log_model = joblib.load(d / "logistic_regression.pkl")
    dt_model = joblib.load(d / "decision_tree.pkl")
    scaler = joblib.load(d / "scalers.pkl")
    label_encoders = joblib.load(d / "label_encoders.pkl")

    feature_names_path = d / "feature_names.pkl"
    try:
        feature_names = joblib.load(feature_names_path)
    except Exception:
        # Fallback for missing or corrupted feature_names.pkl
        from src.preprocessing import FEATURE_COLS
        feature_names = FEATURE_COLS

    return log_model, dt_model, scaler, label_encoders, feature_names


def load_evaluation_metrics(models_dir: Path | str | None = None) -> dict | None:
    """Load evaluation metrics saved by train.py, or None if unavailable.

    Parameters
    ----------
    models_dir : path-like, optional
        Override the default models/ directory.

    Returns
    -------
    dict or None
        Parsed JSON metrics, or None when the file does not exist.
    """
    d = Path(models_dir) if models_dir else MODELS_DIR
    metrics_path = d / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model, X) -> dict:
    """Run inference and return structured prediction results.

    Parameters
    ----------
    model : fitted sklearn estimator
        Binary classifier with predict / predict_proba methods.
    X : array-like, shape (n_samples, n_features)
        Pre-processed feature matrix (scaled for LR, unscaled for DT).

    Returns
    -------
    dict with keys:
        prediction  : list[str]  — 'Churned' | 'Retained' per player
        churn_prob  : np.ndarray — probability of churn (float 0–1)
        risk        : list[str]  — 'Low' | 'Medium' | 'High' risk label
    """
    preds = model.predict(X)
    proba = model.predict_proba(X)

    # Class 1 == Churned (Low engagement); class 0 == Retained
    churn_prob = proba[:, 1]

    prediction_labels = ["Churned" if p == 1 else "Retained" for p in preds]

    risk = [
        "Low" if p < 0.30 else "Medium" if p <= 0.70 else "High"
        for p in churn_prob
    ]

    return {
        "prediction": prediction_labels,
        "churn_prob": churn_prob,
        "risk": risk,
    }
