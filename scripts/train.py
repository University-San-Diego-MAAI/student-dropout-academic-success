"""Training entrypoint: load -> engineer -> train -> save."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dropout.config import get_settings
from dropout.data.loader import load_dataset
from dropout.features.engineer import (
    OHE_COLS,
    ORDINAL_COLS,
    add_derived_features,
    build_preprocessor,
)
from dropout.models.trainer import train_and_save

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

settings = get_settings()

print("Loading dataset...")
df = load_dataset()

print("Engineering features...")
df = add_derived_features(df)

derived = ["approval_rate_s1", "approval_rate_s2", "grade_delta", "financial_risk"]
numeric_cols = [
    c
    for c in df.columns
    if c not in ORDINAL_COLS + OHE_COLS + derived + ["target"]
] + derived

preprocessor = build_preprocessor(numeric_cols)

print("Training models (takes ~5-10 min with Optuna)...")
train_and_save(df, preprocessor, settings.artifact_dir)

print("Done. Artifacts in:", settings.artifact_dir)
