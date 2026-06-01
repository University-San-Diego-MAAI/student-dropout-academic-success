"""Inference wrapper for loading saved artifacts and predicting dropout risk."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class Predictor:
    """Load trained model artifacts and run single-record inference."""

    def __init__(self, model_path: Path, preprocessor_path: Path, le_path: Path) -> None:
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.le = joblib.load(le_path)

    def predict(self, features: dict) -> dict:
        """Run inference on a single record and return structured results."""
        df = pd.DataFrame([features])
        X = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        label = self.le.classes_[pred_idx]
        dropout_prob = float(proba[list(self.le.classes_).index("Dropout")])

        if dropout_prob >= 0.6:
            risk_tier = "High"
        elif dropout_prob >= 0.35:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        return {
            "predicted_outcome": label,
            "dropout_probability": round(dropout_prob, 4),
            "risk_tier": risk_tier,
            "class_probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(self.le.classes_, proba)
            },
        }
