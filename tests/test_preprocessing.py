"""
test_preprocessing.py — Unit tests for src/preprocessing.py.

Tests validate:
  - Schema validation (pass and fail cases)
  - Missing value imputation for numeric and categorical columns
  - Output shape and dtype after full preprocess() pipeline
  - Column ordering is preserved
  - No NaN values survive preprocessing
"""

import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path

from src.preprocessing import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    encode_categoricals,
    impute_missing,
    preprocess,
    validate_schema,
)

MODELS_DIR = Path(__file__).parent.parent / "models"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def label_encoders():
    """Load fitted LabelEncoders from disk."""
    return joblib.load(MODELS_DIR / "label_encoders.pkl")


@pytest.fixture(scope="module")
def scaler():
    """Load fitted StandardScaler from disk."""
    return joblib.load(MODELS_DIR / "scalers.pkl")


@pytest.fixture
def sample_df():
    """Two clean rows covering known categorical values."""
    return pd.DataFrame(
        {
            "PlayerID": [1, 2],
            "Age": [25, 30],
            "Gender": ["Male", "Female"],
            "Location": ["USA", "Europe"],
            "GameGenre": ["Action", "Strategy"],
            "PlayTimeHours": [5.5, 12.0],
            "InGamePurchases": [1, 0],
            "GameDifficulty": ["Medium", "Hard"],
            "SessionsPerWeek": [3, 10],
            "AvgSessionDurationMinutes": [45.0, 90.0],
            "PlayerLevel": [20, 50],
            "AchievementsUnlocked": [5, 25],
        }
    )


@pytest.fixture
def sample_df_with_nan(sample_df):
    """Same two rows but with one numeric and one categorical NaN."""
    df = sample_df.copy()
    df.loc[0, "Age"] = np.nan
    df.loc[1, "Gender"] = np.nan
    return df


# ── validate_schema ───────────────────────────────────────────────────────────

def test_validate_schema_passes_with_all_columns(sample_df):
    assert validate_schema(sample_df) is True


def test_validate_schema_raises_on_missing_column(sample_df):
    bad = sample_df.drop(columns=["Age"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(bad)


def test_validate_schema_raises_on_multiple_missing(sample_df):
    bad = sample_df.drop(columns=["Age", "Gender"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(bad)


# ── impute_missing ────────────────────────────────────────────────────────────

def test_impute_fills_numeric_nan(sample_df_with_nan):
    result = impute_missing(sample_df_with_nan)
    for col in NUMERIC_COLS:
        assert result[col].isna().sum() == 0, f"NaN remains in numeric col: {col}"


def test_impute_fills_categorical_nan(sample_df_with_nan):
    result = impute_missing(sample_df_with_nan)
    for col in CATEGORICAL_COLS:
        assert result[col].isna().sum() == 0, f"NaN remains in categorical col: {col}"


def test_impute_does_not_modify_original(sample_df_with_nan):
    original_nan_count = sample_df_with_nan.isna().sum().sum()
    impute_missing(sample_df_with_nan)
    # Original should be unchanged (function returns a copy)
    assert sample_df_with_nan.isna().sum().sum() == original_nan_count


# ── encode_categoricals ───────────────────────────────────────────────────────

def test_encode_categoricals_produces_integers(sample_df, label_encoders):
    df_clean = sample_df[FEATURE_COLS].copy()
    result = encode_categoricals(df_clean, label_encoders)
    for col in CATEGORICAL_COLS:
        assert pd.api.types.is_integer_dtype(result[col]), f"{col} not integer after encoding"


def test_encode_categoricals_handles_unseen_value(sample_df, label_encoders):
    df_mod = sample_df[FEATURE_COLS].copy()
    df_mod.loc[0, "Gender"] = "Unknown"  # unseen category
    # Should not raise — falls back to first known class
    result = encode_categoricals(df_mod, label_encoders)
    assert result["Gender"].isna().sum() == 0


# ── preprocess (full pipeline) ────────────────────────────────────────────────

def test_preprocess_output_shapes(sample_df, label_encoders, scaler):
    X_scaled, X_encoded = preprocess(sample_df, label_encoders, scaler)
    assert X_scaled.shape == (2, len(FEATURE_COLS))
    assert X_encoded.shape == (2, len(FEATURE_COLS))


def test_preprocess_no_nan_in_scaled_output(sample_df, label_encoders, scaler):
    X_scaled, _ = preprocess(sample_df, label_encoders, scaler)
    assert not np.isnan(X_scaled).any(), "NaN values found in scaled output"


def test_preprocess_column_order_preserved(sample_df, label_encoders, scaler):
    _, X_encoded = preprocess(sample_df, label_encoders, scaler)
    assert list(X_encoded.columns) == FEATURE_COLS


def test_preprocess_with_nan_rows(sample_df_with_nan, label_encoders, scaler):
    """Full pipeline must handle NaN input without error."""
    X_scaled, X_encoded = preprocess(sample_df_with_nan, label_encoders, scaler)
    assert X_scaled.shape[0] == len(sample_df_with_nan)
    assert not np.isnan(X_scaled).any()


def test_preprocess_validates_schema_by_default(scaler, label_encoders):
    bad_df = pd.DataFrame({"Age": [25], "WrongColumn": ["x"]})
    with pytest.raises(ValueError):
        preprocess(bad_df, label_encoders, scaler)


def test_preprocess_skip_validation_flag(label_encoders, scaler):
    """validate=False should bypass schema check (useful for partial data)."""
    # Only FEATURE_COLS — no PlayerID; validate=True would pass anyway
    minimal = pd.DataFrame(
        {
            "Age": [25],
            "Gender": ["Male"],
            "Location": ["USA"],
            "GameGenre": ["Action"],
            "PlayTimeHours": [5.0],
            "InGamePurchases": [1],
            "GameDifficulty": ["Medium"],
            "SessionsPerWeek": [3],
            "AvgSessionDurationMinutes": [45.0],
            "PlayerLevel": [20],
            "AchievementsUnlocked": [5],
        }
    )
    X_scaled, _ = preprocess(minimal, label_encoders, scaler, validate=False)
    assert X_scaled.shape == (1, len(FEATURE_COLS))
