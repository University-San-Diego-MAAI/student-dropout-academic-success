"""Model training pipeline with multiple classifiers and Optuna tuning."""

import logging
from pathlib import Path

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)


def train_and_save(df: pd.DataFrame, preprocessor, artifact_dir: Path) -> None:
    """Train multiple models, pick best by macro-F1, and save artifacts."""
    artifact_dir.mkdir(exist_ok=True)

    le = LabelEncoder()
    y = le.fit_transform(df["target"])
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]

    # 60/20/20 stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # Fit preprocessor on train only
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # SMOTE on train only
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)

    # --- Model 1: Logistic Regression ---
    scaler = StandardScaler()
    X_lr = scaler.fit_transform(X_train_sm)
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    lr.fit(X_lr, y_train_sm)

    # --- Model 2: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_sm, y_train_sm)

    # --- Model 3: XGBoost + Optuna ---
    def xgb_objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )
        model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train_sm, y_train_sm, eval_set=[(X_val_proc, y_val)], verbose=False)
        return f1_score(y_val, model.predict(X_val_proc), average="macro")

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
    best_xgb = xgb.XGBClassifier(
        **study_xgb.best_params,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    best_xgb.fit(X_train_sm, y_train_sm)

    # --- Model 4: LightGBM + Optuna ---
    def lgb_objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            num_leaves=trial.suggest_int("num_leaves", 20, 120),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
        )
        model = LGBMClassifier(
            **params,
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train_sm,
            y_train_sm,
            eval_set=[(X_val_proc, y_val)],
        )
        return f1_score(y_val, model.predict(X_val_proc), average="macro")

    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(lgb_objective, n_trials=50, show_progress_bar=True)
    best_lgb = LGBMClassifier(
        **study_lgb.best_params,
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        verbose=-1,
        random_state=42,
        n_jobs=-1,
    )
    best_lgb.fit(X_train_sm, y_train_sm)

    # --- Model 5: Stacking ---
    meta = LogisticRegression(max_iter=500, random_state=42)
    stack = StackingClassifier(
        estimators=[("rf", rf), ("xgb", best_xgb), ("lgb", best_lgb)],
        final_estimator=meta,
        cv=5,
        n_jobs=-1,
    )
    stack.fit(X_train_sm, y_train_sm)

    # Pick best by macro-F1 on test set
    candidates = [
        ("lr", lr, X_test_proc),
        ("rf", rf, X_test_proc),
        ("xgb", best_xgb, X_test_proc),
        ("lgb", best_lgb, X_test_proc),
        ("stack", stack, X_test_proc),
    ]
    best_name, best_model, _ = max(
        candidates,
        key=lambda t: f1_score(y_test, t[1].predict(t[2]), average="macro"),
    )
    log.info("Best model: %s", best_name)

    # Save artifacts
    joblib.dump(best_model, artifact_dir / "model.pkl")
    joblib.dump(preprocessor, artifact_dir / "preprocessor.pkl")
    joblib.dump(le, artifact_dir / "label_encoder.pkl")
    log.info("Artifacts saved to %s", artifact_dir)
