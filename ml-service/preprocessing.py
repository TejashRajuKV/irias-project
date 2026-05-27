#!/usr/bin/env python3
"""
IR-AIS ML Pipeline — Data Loading & Preprocessing
Handles data cleaning, imputation, feature engineering, and encoding.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

from config import DATA_PATH, MODEL_DIR, TARGET_CLASS, TARGET_REGR, LEAKAGE_COLS


def load_and_preprocess():
    """
    Load raw CSV, clean, encode, and return features + targets.

    Returns
    -------
    X : pd.DataFrame — encoded feature matrix
    y_class : pd.Series — raw classification target
    y_class_encoded : np.ndarray — label-encoded classification target
    y_regr : pd.Series — regression target
    target_encoder : LabelEncoder — fitted target encoder
    label_encoders : dict — column → fitted LabelEncoder
    feature_names : list — ordered feature column names
    """
    print("=" * 60)
    print("IR-AIS ML Training Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Replace "na", "unknown", empty strings with NaN ──
    print("\n  Replacing 'na', 'unknown', '' with NaN...")
    df.replace(["na", "Na", "NA", "unknown", "Unknown", ""], np.nan, inplace=True)

    # ── Impute missing categorical values with Mode ──
    print("  Imputing missing values with mode...")
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # ── Extract Hour_of_Day from Time ──
    print("  Extracting Hour_of_Day from Time column...")
    df["Hour_of_Day"] = df["Time"].apply(lambda t: int(str(t).split(":")[0]))

    # ── Derive Auxiliary Targets ──
    print("  Deriving new auxiliary targets (Pedestrian_involved, Time_category)...")
    if "Pedestrian_movement" in df.columns:
        df["Pedestrian_involved"] = df["Pedestrian_movement"].apply(
            lambda x: 0 if x == "Not a Pedestrian" else 1
        )
        
    def categorize_time(hour):
        if not pd.isna(hour):
            h = int(hour)
            if 7 <= h <= 10: return "Rush_Morning"
            elif 11 <= h <= 15: return "Midday"
            elif 16 <= h <= 19: return "Rush_Evening"
            else: return "Night"
        return "Unknown"
        
    df["Time_category"] = df["Hour_of_Day"].apply(categorize_time)
    df.drop(columns=["Time"], inplace=True)

    # ── Drop leakage columns ──
    print(f"  Dropping leakage columns: {LEAKAGE_COLS}")
    df.drop(columns=LEAKAGE_COLS, inplace=True, errors="ignore")

    # ── Separate targets ──
    y_class = df[TARGET_CLASS].copy()
    y_regr = df[TARGET_REGR].copy()
    df.drop(columns=[TARGET_CLASS, TARGET_REGR], inplace=True)

    # ── Encode target for classification ──
    print("  Label encoding classification target...")
    target_encoder = LabelEncoder()
    y_class_encoded = target_encoder.fit_transform(y_class)
    print(f"  Target classes: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

    # ── Label encode all categorical features ──
    print("  Label encoding categorical features...")
    label_encoders = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    X = df
    feature_names = list(X.columns)

    print(f"  Final features: {len(feature_names)} columns")
    print(f"  Feature names: {feature_names}")

    # ── Save encoders & feature names ──
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.pkl"))

    return X, y_class, y_class_encoded, y_regr, target_encoder, label_encoders, feature_names
