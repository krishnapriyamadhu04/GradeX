# preprocessing.py
# ─────────────────────────────────────────────────────────────────
# Handles all data-preparation steps before training.
# Reused by both train_model.py (offline) and app.py (live inference).
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Column definitions ───────────────────────────────────────────
FEATURE_COLS = [
    "internal_exam",           # Internal Exam score       (0–50)
    "assignment",              # Assignment score           (0–15)
    "lab_practical",           # Lab / Practical score      (0–100)
    "attendance",              # Attendance marks           (0–10)
    "previous_semester_cgpa",  # Previous Semester CGPA     (0–10)
]
TARGET_COL = "current_semester_cgpa"   # Current Semester CGPA (0–10)


def load_data(filepath: str = "data/student_scores.csv") -> pd.DataFrame:
    """
    Load the CSV dataset from disk and return it as a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all columns intact.
    """
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """
    Print a quick report of missing values per column.
    In a real project you would decide here whether to drop or impute them.
    """
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[check_missing_values] No missing values found. OK")
    else:
        print("[check_missing_values] Missing values detected:")
        print(missing[missing > 0])


def split_features_target(df: pd.DataFrame):
    """
    Separate the DataFrame into features (X) and target (y).

    Returns
    -------
    X : pd.DataFrame  – the five input features
    y : pd.Series     – current_semester_cgpa column
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and test sets.

    80 % → training   (model learns from this)
    20 % → test       (model is evaluated on unseen data)

    Parameters
    ----------
    test_size    : fraction held out for testing (default 0.20)
    random_state : seed so the split is reproducible every run
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[split_train_test] Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standardise features so they all live on the same numeric scale.

    StandardScaler transforms each feature to mean=0, std=1.
    This matters for Linear Regression; Random Forest does not require
    it, but a consistent pipeline avoids mistakes.

    IMPORTANT: the scaler is fitted ONLY on training data, then applied
    (transform only) to both train and test. Fitting on test data would
    cause data leakage — the model would see test-set statistics.

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : fitted StandardScaler (saved so app.py can reuse it)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # learn + scale train
    X_test_scaled  = scaler.transform(X_test)         # scale test with same params
    print("[scale_features] Features standardised (mean=0, std=1).")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath: str = "data/student_scores.csv"):
    """
    Convenience function: runs the full preprocessing pipeline in one call.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    df = load_data(filepath)
    check_missing_values(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
