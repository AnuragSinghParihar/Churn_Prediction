"""
preprocessing.py — Reusable preprocessing pipeline for Player Churn Prediction.

Expected input schema (all columns required):
  Age                      : int   — player age in years
  Gender                   : str   — 'Male' | 'Female'
  Location                 : str   — 'USA' | 'Europe' | 'Asia' | 'Other'
  GameGenre                : str   — 'Action' | 'RPG' | 'Simulation' | 'Sports' | 'Strategy'
  PlayTimeHours            : float — total hours played
  InGamePurchases          : int   — 0 (no) or 1 (yes)
  GameDifficulty           : str   — 'Easy' | 'Medium' | 'Hard'
  SessionsPerWeek          : int   — average sessions per week
  AvgSessionDurationMinutes: float — average session length in minutes
  PlayerLevel              : int   — current player level
  AchievementsUnlocked     : int   — total achievements earned
"""

import numpy as np
import pandas as pd

# ── Schema constants ─────────────────────────────────────────────────────────

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
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]


# ── Individual steps ─────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> bool:
    """Raise ValueError if any required column is missing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.

    Returns
    -------
    bool
        True when all required columns are present.

    Raises
    ------
    ValueError
        Lists every missing column name.
    """
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return True


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in-place using mean (numeric) or mode (categorical).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, may contain NaN values.

    Returns
    -------
    pd.DataFrame
        Copy with no NaN values in feature columns.
    """
    df_out = df.copy()
    for col in NUMERIC_COLS:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(df_out[col].mean())
    for col in CATEGORICAL_COLS:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(df_out[col].mode()[0])
    return df_out


def encode_categoricals(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """Apply fitted LabelEncoders to categorical columns.

    Unseen category values are replaced with the first known class before
    encoding to avoid transform errors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all CATEGORICAL_COLS present as string columns.
    label_encoders : dict[str, LabelEncoder]
        Mapping of column name → fitted sklearn LabelEncoder.

    Returns
    -------
    pd.DataFrame
        Copy with categorical columns replaced by integer codes.
    """
    df_out = df.copy()
    for col in CATEGORICAL_COLS:
        if col not in df_out.columns:
            continue
        le = label_encoders.get(col)
        if le is not None:
            # Map unseen categories to the first known class (fallback)
            df_out.loc[~df_out[col].isin(le.classes_), col] = le.classes_[0]
            df_out[col] = le.transform(df_out[col])
        else:
            df_out[col] = pd.factorize(df_out[col])[0]
    return df_out


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    label_encoders: dict,
    scaler,
    validate: bool = True,
):
    """End-to-end preprocessing: validate → impute → select → encode → scale.

    Parameters
    ----------
    df : pd.DataFrame
        Raw player-behavior data with at least the columns in FEATURE_COLS.
    label_encoders : dict
        Fitted LabelEncoders for categorical columns (loaded from models/).
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler (loaded from models/).
    validate : bool, optional
        Whether to run schema validation. Default True.

    Returns
    -------
    X_scaled : np.ndarray, shape (n_samples, 11)
        Scaled feature matrix — use as input for Logistic Regression.
    X_encoded : pd.DataFrame, shape (n_samples, 11)
        Encoded (but unscaled) feature DataFrame — use as input for Decision Tree.
    """
    if validate:
        validate_schema(df)

    df_clean = impute_missing(df)
    df_selected = df_clean[FEATURE_COLS].copy()
    df_encoded = encode_categoricals(df_selected, label_encoders)
    X_scaled = scaler.transform(df_encoded)

    return X_scaled, df_encoded
