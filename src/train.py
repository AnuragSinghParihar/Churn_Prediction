"""
train.py — Full training pipeline for Player Churn Prediction.

Pipeline:
  1. Load raw data
  2. Binarise target: Low engagement → 1 (Churn), others → 0 (Retained)
  3. Encode categoricals with LabelEncoder, scale with StandardScaler
  4. Cross-validate with StratifiedKFold (k=5)
  5. Evaluate on hold-out test set (80/20 split)
  6. Save models, preprocessing artefacts, metrics, and feature importances

Usage:
    python -m src.train
    python -m src.train --data data/custom.csv --models models/
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ── Default paths ─────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
DATA_PATH = _ROOT / "data" / "online_gaming_behavior_dataset.csv"
MODELS_DIR = _ROOT / "models"

# ── Schema constants ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "Age",
    "Gender",
    "Location",
    "GameGenre",
    "PlayTimeHours",
    "InGamePurchases",
    "GameDifficulty",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "PlayerLevel",
    "AchievementsUnlocked",
]
CATEGORICAL_COLS = ["Gender", "Location", "GameGenre", "GameDifficulty"]
TARGET_COL = "EngagementLevel"
CHURN_CLASS = "Low"  # Low engagement → Churn = 1


# ── Data preparation ──────────────────────────────────────────────────────────

def load_data(data_path=None):
    """Load raw CSV data.

    Parameters
    ----------
    data_path : path-like, optional
        Override the default data path.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns.
    """
    path = Path(data_path) if data_path else DATA_PATH
    return pd.read_csv(path)


def prepare_features(df):
    """Encode categoricals and binarise the target.

    The target is binarised as:
      1 → Low engagement (Churn)
      0 → Medium or High engagement (Retained)

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with FEATURE_COLS + TARGET_COL present.

    Returns
    -------
    X : pd.DataFrame
        Encoded feature matrix (n_samples × 11).
    y : np.ndarray
        Binary target vector (0 / 1).
    label_encoders : dict[str, LabelEncoder]
        Fitted encoders for each categorical column.
    """
    label_encoders = {}
    df_enc = df[FEATURE_COLS].copy()

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        label_encoders[col] = le

    y = (df[TARGET_COL] == CHURN_CLASS).astype(int).values
    return df_enc, y, label_encoders


# ── Evaluation helpers ────────────────────────────────────────────────────────

def cross_validate_model(model, X, y, cv=5):
    """Run StratifiedKFold cross-validation and return aggregated metrics.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted classifier.
    X : array-like
        Feature matrix.
    y : array-like
        Binary target vector.
    cv : int
        Number of folds (default 5).

    Returns
    -------
    dict
        Mean and std for accuracy, F1, and ROC-AUC across folds.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=True,
    )
    return {
        "cv_folds": cv,
        "cv_accuracy_mean": float(results["test_accuracy"].mean()),
        "cv_accuracy_std": float(results["test_accuracy"].std()),
        "cv_f1_mean": float(results["test_f1"].mean()),
        "cv_f1_std": float(results["test_f1"].std()),
        "cv_roc_auc_mean": float(results["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(results["test_roc_auc"].std()),
    }


def evaluate_on_test(model, X_test, y_test):
    """Evaluate a fitted model on a held-out test set.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : array-like
    y_test : array-like

    Returns
    -------
    dict
        accuracy, roc_auc, and the full classification_report as a nested dict.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Retained", "Churned"],
            output_dict=True,
        ),
    }


def get_feature_importances(dt_model, feature_names):
    """Extract and sort Decision Tree feature importances.

    Parameters
    ----------
    dt_model : fitted DecisionTreeClassifier
    feature_names : list[str]

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'importance'], sorted descending.
    """
    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": dt_model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ── Main training function ────────────────────────────────────────────────────

def train_and_save(data_path=None, models_dir=None):
    """Run the full train / evaluate / save pipeline.

    Parameters
    ----------
    data_path : path-like, optional
        Path to the raw CSV dataset.
    models_dir : path-like, optional
        Directory where artefacts are saved.

    Returns
    -------
    dict
        Evaluation metrics for both models.
    """
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & prepare --------------------------------------------------------
    print("Loading data ...")
    df = load_data(data_path)
    print(f"  {len(df):,} rows x {df.shape[1]} columns")

    print("Encoding features and binarising target ...")
    X, y, label_encoders = prepare_features(df)
    print(f"  Churn (Low engagement) ratio: {y.mean():.2%}")

    # 2. Train / test split ----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Scale (fit on train only) ---------------------------------------------
    print("Fitting scaler ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Logistic Regression ---------------------------------------------------
    print("\nTraining Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, random_state=42)

    print("  Cross-validating (5-fold) ...")
    lr_cv = cross_validate_model(
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_scaled,
        y_train,
    )

    lr.fit(X_train_scaled, y_train)
    lr_test = evaluate_on_test(lr, X_test_scaled, y_test)
    lr_metrics = {**lr_cv, **lr_test}

    print(f"  CV Accuracy : {lr_cv['cv_accuracy_mean']:.4f} +/- {lr_cv['cv_accuracy_std']:.4f}")
    print(f"  CV ROC-AUC  : {lr_cv['cv_roc_auc_mean']:.4f}")
    print(f"  Test Accuracy: {lr_test['test_accuracy']:.4f}")
    print(f"  Test ROC-AUC : {lr_test['test_roc_auc']:.4f}")

    # 5. Decision Tree ---------------------------------------------------------
    print("\nTraining Decision Tree ...")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)

    print("  Cross-validating (5-fold) ...")
    dt_cv = cross_validate_model(
        DecisionTreeClassifier(max_depth=5, random_state=42),
        X_train.values,
        y_train,
    )

    dt.fit(X_train, y_train)
    dt_test = evaluate_on_test(dt, X_test, y_test)
    dt_metrics = {**dt_cv, **dt_test}

    print(f"  CV Accuracy : {dt_cv['cv_accuracy_mean']:.4f} +/- {dt_cv['cv_accuracy_std']:.4f}")
    print(f"  CV ROC-AUC  : {dt_cv['cv_roc_auc_mean']:.4f}")
    print(f"  Test Accuracy: {dt_test['test_accuracy']:.4f}")
    print(f"  Test ROC-AUC : {dt_test['test_roc_auc']:.4f}")

    # 6. Feature importances ---------------------------------------------------
    feature_importance = get_feature_importances(dt, list(X.columns))
    print("\nTop 5 features (Decision Tree):")
    print(feature_importance.head().to_string(index=False))

    # 7. Save artefacts --------------------------------------------------------
    print("\nSaving artefacts ...")
    joblib.dump(lr, models_dir / "logistic_regression.pkl")
    joblib.dump(dt, models_dir / "decision_tree.pkl")
    joblib.dump(scaler, models_dir / "scalers.pkl")
    joblib.dump(label_encoders, models_dir / "label_encoders.pkl")
    joblib.dump(list(X.columns), models_dir / "feature_names.pkl")

    all_metrics = {
        "logistic_regression": lr_metrics,
        "decision_tree": dt_metrics,
        "feature_importance": feature_importance.to_dict(orient="records"),
        "churn_class": CHURN_CLASS,
        "target_col": TARGET_COL,
    }
    with open(models_dir / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("  Saved: logistic_regression.pkl, decision_tree.pkl")
    print("  Saved: scalers.pkl, label_encoders.pkl, feature_names.pkl")
    print("  Saved: evaluation_metrics.json")
    print("\nDone.")

    return all_metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train Player Churn Prediction models.")
    p.add_argument("--data", default=None, help="Path to CSV dataset")
    p.add_argument("--models", default=None, help="Output directory for artefacts")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_and_save(data_path=args.data, models_dir=args.models)
