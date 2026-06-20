# 🎓 Student Dropout & Academic Success Prediction

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-ff4b4b?logo=streamlit)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-f7931e?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-orange?logo=xgboost)](https://xgboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-green)](https://lightgbm.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-4.1+-purple)](https://optuna.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.46+-red)](https://shap.readthedocs.io)
[![Ruff](https://img.shields.io/badge/Ruff-linter-black?logo=ruff)](https://docs.astral.sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🏫 **AAI-510: Machine Learning for Artificial Intelligence** — University of San Diego  
> 🔬 A production-ready multiclass ML pipeline predicting student outcomes: **Dropout** | **Enrolled** | **Graduate**

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📊 Dataset](#-dataset)
- [🚀 Quick Start](#-quick-start)
- [🛠️ Installation](#️-installation)
- [🧠 Model Training](#-model-training)
- [🌐 API Usage](#-api-usage)
- [🖥️ Streamlit UI](#️-streamlit-ui)
- [📓 Jupyter Notebook](#-jupyter-notebook)
- [🔐 Environment Variables](#-environment-variables)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing & Linting](#-testing--linting)
- [📚 Course Information](#-course-information)
- [📝 License](#-license)

---

## ✨ Features

| Feature | Description | Tech |
|---------|-------------|------|
| 🔍 **Auto Dataset Fetch** | Downloads UCI dataset (ID 697) automatically or loads from local cache | `ucimlrepo` |
| 🧬 **Feature Engineering** | 4 derived features: approval rates, grade delta, financial risk | `pandas` |
| 🎨 **EDA & Visualization** | Phi-k correlations, distribution plots, SHAP explainability | `seaborn`, `matplotlib`, `shap` |
| 🤖 **5-Model Ensemble** | Logistic Regression, Random Forest, XGBoost, LightGBM + Stacking | `sklearn`, `xgboost`, `lightgbm` |
| ⚡ **Bayesian Optimization** | 50-trial Optuna tuning for XGBoost & LightGBM | `optuna` |
| ⚖️ **SMOTE Resampling** | Addresses class imbalance on training data only | `imbalanced-learn` |
| 🏆 **Best Model Selection** | Picks winner by macro-F1 on held-out test set | — |
| 🔮 **Risk Tier Classification** | Low / Medium / High dropout probability tiers | — |
| 🚀 **FastAPI REST API** | Real-time inference endpoint with Pydantic validation | `fastapi`, `uvicorn` |
| 💻 **Streamlit Dashboard** | Interactive web UI for academic advisors | `streamlit` |
| 📓 **Jupyter Notebook** | Complete 8-section analysis for university submission | `jupyter` |
| 🔒 **Reproducible** | Fixed random seeds, cached data, version-pinned dependencies | `uv` |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   UCI Dataset   │────▶│  Feature Eng.    │────▶│ 5-Model Train   │
│   (ID 697)      │     │ + SMOTE + Optuna │     │ + Best Select   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                              ┌──────────────────────────────────────┐
                              │        Saved Artifacts               │
                              │  artifacts/model.pkl                 │
                              │  artifacts/preprocessor.pkl          │
                              │  artifacts/label_encoder.pkl         │
                              └──────────────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴─────────────────────────────────────┐
                    │                                                                             │
                    ▼                                                                             ▼
          ┌─────────────────┐                                                           ┌─────────────────┐
          │   FastAPI       │◄──── POST /predict ────│──── JSON Request               │   Jupyter       │
          │   (Port 8000)   │                        │    (36 features)               │   Notebook      │
          │                 │                        │                                │   (EDA + Model) │
          │  /health        │                        │                                │                 │
          │  /predict       │──── JSON Response ─────│                                │                 │
          │  /docs (Swagger)│    (risk tier + probs)│                                │                 │
          └─────────────────┘                        │                                └─────────────────┘
                    │                                │
                    ▼                                │
          ┌─────────────────┐                       │
          │   Streamlit UI  │◄──────────────────────┘
          │   (Port 8501)   │    Form + Visualization
          │                 │
          │  Risk Badge     │
          │  Probability Bar│
          │  Class Breakdown│
          └─────────────────┘
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [UCI ML Repository — ID 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| **Records** | 4,424 students |
| **Features** | 36 (demographic, socioeconomic, academic, macroeconomic) |
| **Target Classes** | `Dropout` (32.1%), `Enrolled` (17.9%), `Graduate` (50.0%) |
| **Missing Values** | 0 |
| **Data Cached At** | `data/raw/features.csv`, `data/raw/targets.csv` |

---

## 🚀 Quick Start

### Prerequisites

- 🐍 Python 3.12+
- 📦 [uv](https://docs.astral.sh/uv/) package manager (recommended)

### 1️⃣ Clone & Enter the Project

```bash
git clone <your-repo-url>
cd student-dropout-project
```

### 2️⃣ Install Dependencies

```bash
uv sync
```

> 💡 If you encounter workspace resolution issues, temporarily move the parent `pyproject.toml` in the parent directory, run `uv sync`, then restore it.

### 3️⃣ Train the Model (One-Time, ~10–15 min)

```bash
uv run python scripts/train.py
```

Output:
```
Loading dataset...
Engineering features...
Training models (takes ~5-10 min with Optuna)...
Best model: stack
Artifacts saved to artifacts
Done. Artifacts in: artifacts
```

This generates:
- `artifacts/model.pkl` — Best model (StackingClassifier)
- `artifacts/preprocessor.pkl` — Sklearn ColumnTransformer
- `artifacts/label_encoder.pkl` — Target label encoder

### 4️⃣ Start the FastAPI Server

```bash
uv run uvicorn dropout.api.main:app --reload --port 8000
```

Visit:
- 📖 **API Docs**: http://localhost:8000/docs
- 🔍 **Health Check**: http://localhost:8000/health

### 5️⃣ Launch the Streamlit UI (New Terminal)

```bash
uv run streamlit run src/dropout/ui/app.py
```

Open: http://localhost:8501

### 6️⃣ Run the Jupyter Notebook

```bash
uv run jupyter notebook notebooks/student_dropout_analysis.ipynb
```

---

## 🛠️ Installation

### Using `uv` (Recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies from pyproject.toml
uv sync

# Install dev dependencies (pytest, ruff, httpx)
uv pip install -e ".[dev]"
```

### Using `pip` (Alternative)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## 🧠 Model Training

The training pipeline implements a complete ML workflow:

| Step | Details |
|------|---------|
| **Data Split** | 60% train / 20% validation / 20% test (stratified) |
| **Preprocessing** | OrdinalEncoder (7 cols) + OneHotEncoder (2 cols) + passthrough numeric |
| **SMOTE** | Applied **only** on training set to balance classes |
| **Models Trained** | Logistic Regression, Random Forest, XGBoost (Optuna 50 trials), LightGBM (Optuna 50 trials), Stacking Ensemble |
| **Selection Metric** | Macro-F1 on held-out test set |
| **Best Model** | StackingClassifier (RF + XGBoost + LightGBM → LogisticRegression meta) |

### Re-train from Scratch

```bash
rm -rf artifacts/*
uv run python scripts/train.py
```

---

## 🌐 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

### Predict Dropout Risk

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "marital_status": 1,
    "gender": 1,
    "age_at_enrollment": 20,
    "nationality": 1,
    "international": 0,
    "application_mode": 1,
    "application_order": 1,
    "course": 1,
    "daytime_evening_attendance": 1,
    "previous_qualification": 1,
    "previous_qualification_grade": 130.0,
    "admission_grade": 130.0,
    "displaced": 0,
    "mothers_qualification": 1,
    "fathers_qualification": 1,
    "mothers_occupation": 1,
    "fathers_occupation": 1,
    "educational_special_needs": 0,
    "debtor": 0,
    "tuition_fees_up_to_date": 1,
    "scholarship_holder": 0,
    "curricular_units_1st_sem_credited": 0,
    "curricular_units_1st_sem_enrolled": 6,
    "curricular_units_1st_sem_evaluations": 6,
    "curricular_units_1st_sem_approved": 6,
    "curricular_units_1st_sem_grade": 12.0,
    "curricular_units_1st_sem_without_evaluations": 0,
    "curricular_units_2nd_sem_credited": 0,
    "curricular_units_2nd_sem_enrolled": 6,
    "curricular_units_2nd_sem_evaluations": 6,
    "curricular_units_2nd_sem_approved": 6,
    "curricular_units_2nd_sem_grade": 13.0,
    "curricular_units_2nd_sem_without_evaluations": 0,
    "unemployment_rate": 10.0,
    "inflation_rate": 1.5,
    "gdp": 1.0
  }'
```

Response:
```json
{
  "predicted_outcome": "Graduate",
  "dropout_probability": 0.0307,
  "risk_tier": "Low",
  "class_probabilities": {
    "Dropout": 0.0307,
    "Enrolled": 0.0436,
    "Graduate": 0.9257
  }
}
```

### Risk Tier Thresholds

| Tier | Dropout Probability | Action |
|------|---------------------|--------|
| 🟢 **Low** | < 35% | Monitor |
| 🟡 **Medium** | 35% – 60% | Early intervention |
| 🔴 **High** | ≥ 60% | Immediate support |

---

## 🖥️ Streamlit UI

The Streamlit dashboard provides an interactive form for all 36 student features, with:

- 🎨 Color-coded risk badges (Low = green, Medium = yellow, High = red)
- 📊 Progress bars for class probabilities
- 📋 Organized sections: Demographics, Academic Background, Socioeconomic, Semester 1, Semester 2, Macroeconomic

### Start the UI

```bash
# Terminal 1: Start API
uv run uvicorn src.dropout.api.main:app --port 8000

# Terminal 2: Start UI
uv run streamlit run src/dropout/ui/app.py
```

---

## 📓 Jupyter Notebook

The comprehensive analysis notebook covers all 8 required university sections:

1. 📌 Problem Statement & Justification
2. 📊 Data Understanding (EDA) — graphical & non-graphical
3. 🧹 Data Preparation & Feature Engineering
4. 🎯 Feature Selection
5. 🤖 Modeling — 5 models, tuning, ensembles
6. 📈 Evaluation — metrics, confusion matrix, SHAP
7. 🚀 Deployment — FastAPI + Streamlit discussion
8. 💡 Discussion & Conclusions

### Run the Notebook

```bash
uv run jupyter notebook notebooks/student_dropout_analysis.ipynb
```

> ✅ All cells are pre-executed with outputs — ready for submission!

---

## 🔐 Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTIFACT_DIR` | `artifacts` | Directory for saved models |
| `MODEL_PATH` | `artifacts/model.pkl` | Trained model path |
| `PREPROCESSOR_PATH` | `artifacts/preprocessor.pkl` | Preprocessor pipeline path |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

No API keys or secrets are required — the dataset is public and fetched from UCI ML Repository.

---

## 📁 Project Structure

```
student-dropout-project/
├── 📄 README.md                          # This file
├── 📄 pyproject.toml                     # Dependencies & project config
├── 📄 .env.example                       # Environment template
├── 📄 .gitignore                         # Git exclusions
├── 📄 project_implementation.md         # Detailed implementation spec
│
├── 📁 artifacts/                         # 🗂️ Saved model files (gitignored)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
│
├── 📁 data/
│   └── 📁 raw/                           # 📊 Cached dataset (gitignored)
│       ├── features.csv
│       └── targets.csv
│
├── 📁 notebooks/
│   └── student_dropout_analysis.ipynb   # 📓 Full analysis (pre-executed)
│
├── 📁 scripts/
│   └── train.py                          # 🏋️ Training entrypoint
│
├── 📁 src/
│   └── 📁 dropout/
│       ├── 📄 config.py                  # ⚙️ Pydantic settings
│       ├── 📁 data/
│       │   └── loader.py                 # 📥 UCI dataset fetcher
│       ├── 📁 features/
│       │   └── engineer.py              # 🧬 Feature engineering
│       ├── 📁 models/
│       │   ├── trainer.py               # 🤖 5-model training pipeline
│       │   └── predictor.py             # 🔮 Inference wrapper
│       ├── 📁 api/
│       │   ├── main.py                  # 🚀 FastAPI app factory
│       │   ├── routes.py                # 🛣️ /health, /predict endpoints
│       │   └── schemas.py               # 📐 Pydantic request/response models
│       └── 📁 ui/
│           └── app.py                    # 💻 Streamlit dashboard
│
└── 📄 main.py                            # Legacy placeholder
```

---

## 🧪 Testing & Linting

### Code Quality

```bash
# Lint check
uv run ruff check src/ scripts/

# Auto-fix issues
uv run ruff check src/ scripts/ --fix

# Format code
uv run ruff format src/ scripts/
```

### API Testing

```bash
# Start the server
uv run uvicorn src.dropout.api.main:app --port 8000

# In another terminal, run the example curl commands above
# Or use the notebook's deployment section for interactive testing
```

---

## 📚 Course Information

| | |
|:---|:---|
| **Course** | AAI-510: Machine Learning for Artificial Intelligence |
| **Institution** | University of San Diego |
| **Dataset** | UCI ML Repository — [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| **Problem Type** | Multiclass Classification (3 classes) |
| **Target Metric** | Macro-F1 Score ≥ 0.70 |
| **Best Result** | **Macro-F1 = 0.7087** (Stacking Ensemble) |

---

## 📝 License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.

---

## 🤝 Acknowledgments

- 📊 Dataset provided by [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- ⚡ Built with [FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://streamlit.io/), and [scikit-learn](https://scikit-learn.org/)
- 🎓 Developed for AAI-510 at University of San Diego

---

<p align="center">
  Made with ❤️ for better student retention
</p>
