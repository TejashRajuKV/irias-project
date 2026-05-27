#!/usr/bin/env python3
"""
IR-AIS Prediction Helper
Loads trained models and makes predictions on new data.
"""

import os
import numpy as np
import pandas as pd
import joblib

from config import MODEL_DIR


def _load_artifacts():
    """Load all saved artifacts needed for prediction."""
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
    classifier = joblib.load(os.path.join(MODEL_DIR, "best_classifier.pkl"))
    regressor = joblib.load(os.path.join(MODEL_DIR, "best_regressor.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return label_encoders, feature_names, target_encoder, classifier, regressor, scaler


def _transform_features(features_dict: dict, label_encoders: dict, feature_names: list) -> pd.DataFrame:
    """
    Transform raw feature dict into a numeric DataFrame using saved LabelEncoders.
    Handles unknown categories by mapping to the most frequent (index 0).
    """
    row = {}
    for col in feature_names:
        val = features_dict.get(col)
        if col in label_encoders:
            le = label_encoders[col]
            
            # Map user-friendly labels back to raw values for specific columns
            if col == "Defect_of_vehicle":
                if val == "Defective": val = "5"
                # "No defect" is already "No defect"
            
            try:
                encoded = le.transform([val])[0]
            except (ValueError, TypeError):
                # Unknown category → map to most frequent class (index 0)
                encoded = 0
            row[col] = encoded
        else:
            # Numeric feature — pass through
            row[col] = float(val) if val is not None else 0.0
    return pd.DataFrame([row], columns=feature_names)


def classify(features_dict: dict) -> dict:
    """
    Takes a dict of feature values, transforms using saved encoders,
    and returns a prediction with probabilities.

    Parameters
    ----------
    features_dict : dict
        Feature names → values. Must contain all feature columns used in training.
        For the Time column, provide the raw time string; it will not be used
        (Hour_of_Day is expected instead, extracted during training).
        The keys should match the feature_names used during training.

    Returns
    -------
    dict with keys:
        - prediction: str, the predicted severity label
        - prediction_encoded: int, the encoded class
        - probabilities: dict mapping class labels to probabilities
        - confidence: float, probability of the predicted class
    """
    label_encoders, feature_names, target_encoder, classifier, _, scaler = _load_artifacts()

    X_df = _transform_features(features_dict, label_encoders, feature_names)
    X_scaled = scaler.transform(X_df)
    
    y_pred_encoded = classifier.predict(X_scaled)[0]
    prediction_label = target_encoder.inverse_transform([y_pred_encoded])[0]

    # Probabilities
    try:
        y_proba = classifier.predict_proba(X_scaled)[0]
        class_labels = target_encoder.inverse_transform(np.arange(len(y_proba)))
        probabilities = {str(label): round(float(prob), 4) for label, prob in zip(class_labels, y_proba)}
        confidence = round(float(max(y_proba)), 4)
    except Exception:
        probabilities = None
        confidence = None

    return {
        "prediction": str(prediction_label),
        "prediction_encoded": int(y_pred_encoded),
        "probabilities": probabilities,
        "confidence": confidence,
    }


def regress(features_dict: dict) -> dict:
    """
    Takes a dict of feature values, transforms using saved encoders,
    and returns the predicted casualty count.

    Parameters
    ----------
    features_dict : dict
        Feature names → values.

    Returns
    -------
    dict with keys:
        - prediction: float, predicted number of casualties
        - prediction_rounded: int, rounded to nearest integer
    """
    label_encoders, feature_names, target_encoder, classifier, regressor, scaler = _load_artifacts()

    X_df = _transform_features(features_dict, label_encoders, feature_names)
    X_scaled = scaler.transform(X_df)
    
    y_pred = regressor.predict(X_scaled)[0]

    return {
        "prediction": round(float(y_pred), 4),
        "prediction_rounded": int(round(y_pred)),
    }


def predict_auxiliary(features_dict: dict) -> dict:
    """
    Predict all auxiliary targets dynamically.
    """
    from config import AUXILIARY_TASKS
    
    label_encoders, feature_names, _, _, _, _ = _load_artifacts()
    X_df_base = _transform_features(features_dict, label_encoders, feature_names)
    
    aux_results = {}
    
    for task_name, target_col in AUXILIARY_TASKS.items():
        safe_name = task_name.replace(" ", "_").lower()
        model_path = os.path.join(MODEL_DIR, f"best_aux_{safe_name}.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"scaler_aux_{safe_name}.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            continue
            
        aux_model = joblib.load(model_path)
        aux_scaler = joblib.load(scaler_path)
        
        # Prepare specific feature set
        X_task = X_df_base.copy()
        if target_col in X_task.columns:
            X_task.drop(columns=[target_col], inplace=True)
            
        if target_col == "Pedestrian_involved" and "Pedestrian_movement" in X_task.columns:
            X_task.drop(columns=["Pedestrian_movement"], inplace=True)
        if target_col == "Time_category" and "Hour_of_Day" in X_task.columns:
            X_task.drop(columns=["Hour_of_Day"], inplace=True)
            
        try:
            X_scaled = aux_scaler.transform(X_task)
            
            # Note: For XGBoost, targets were encoded integers. For RF, they were arbitrary strings. 
            # predict.py does not have target_encoders for auxiliary tasks (since they used label_encoder globally before splitting)
            # Fortunately, label_encoders covers all features from preprocessing.
            y_pred = aux_model.predict(X_scaled)[0]
            
            # If the model emits raw string categories, we're good. If it emits ints, we must decode.
            if isinstance(y_pred, (int, np.integer)):
                if target_col in label_encoders:
                    # we can inverse transform it
                    y_pred = label_encoders[target_col].inverse_transform([y_pred])[0]
                elif target_col == "Pedestrian_involved":
                    y_pred = "Yes" if y_pred == 1 else "No"
                    
            # Basic probability
            confidence = None
            if hasattr(aux_model, "predict_proba"):
                try:
                    proba = aux_model.predict_proba(X_scaled)[0]
                    confidence = float(max(proba))
                except:
                    pass
            
            aux_results[task_name] = {
                "prediction": str(y_pred),
                "confidence": confidence
            }
        except Exception as e:
            aux_results[task_name] = {"error": str(e)}

    return aux_results


# ── CLI helper for quick testing ──────────────────────────────────────────────
if __name__ == "__main__":
    print("IR-AIS Prediction Helper")
    print("Usage:")
    print("  from predict import classify, regress")
    print("  result = classify({...})")
    print("  result = regress({...})")
    print()
    print("Example features:")
    print("  features = {")
    print("      'Day_of_week': 'Monday',")
    print("      'Age_band_of_driver': '18-30',")
    print("      'Sex_of_driver': 'Male',")
    print("      'Hour_of_Day': 17,")
    print("      ...}")
