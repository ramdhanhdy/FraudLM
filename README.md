# FraudLM – Fraud Detection with LLM-Powered Explainability

**Status**: Early-stage work in progress (WIP) / experimental.

FraudLM is an experimental project exploring how to combine classical fraud detection models with **Large Language Models (LLMs)** to deliver human-readable explanations. The sections below describe the **intended** architecture and experiments; many components are not yet implemented or are only partial prototypes and may change significantly.

---

## Features

> Note: This section describes the **target** feature set; many items are still **planned** or only partially implemented.

### Core System
- **End-to-end pipeline (in progress / planned)**  
  From raw transaction data → experiments → deployed model + LLM-powered dashboard.

- **Fraud detection models (in progress / planned)**  
  - Baselines (planned): Logistic Regression, XGBoost.
  - Extended experiments (planned): CatBoost, FT-Transformer, GraphSAGE.

- **LLM-powered explainability (planned; the "LM" in FraudLM)**  
  - Intended use of **SHAP** to provide technical feature importance.
  - Planned **LLM** component to translate SHAP outputs into natural language explanations tailored for:
    - **Analysts**: technical, precise explanations with key risk drivers.
    - **Customers**: simple, reassuring messages about flagged transactions.
  - Dual-audience explanations intended to be surfaced in real-time via the dashboard.

- **Experiment tracking with MLflow (planned)**  
  - Experiments are planned to be logged with params, metrics, and artifacts.
  - Side-by-side comparison of models, feature sets, and training setups (planned).

- **Application layer (planned)**  
  - Planned FastAPI backend with `/predict` and `/explain` endpoints.
  - Planned Streamlit dashboard with analyst and customer explanation views.

### Research Extensions
- **CTGAN-based synthetic data generation**  
  - Study rare fraud patterns and privacy-preserving data sharing.
  - Experiments comparing models trained on real vs synthetic vs mixed data.

---

## Architecture Overview

> Note: This is the **intended** architecture. The current codebase only partially implements these components and the structure may change.

The project is split into three main layers:

- **Experiments (notebooks + src)**  
  Jupyter notebooks orchestrate experiments. Reusable data, feature, model, and evaluation code lives under `src/`.

- **Model serving (API)**  
  A FastAPI service loads the fraud detection model and SHAP explainer, then exposes:
  - `/predict` – returns fraud probability + SHAP feature importance.
  - `/explain` – calls an LLM to generate natural language explanations for analysts or customers.

- **Analytics UI (dashboard)**  
  A Streamlit app calls the API and provides:
  - Real-time fraud scoring with dual-view explanations (analyst vs customer).
  - SHAP visualizations and model performance summaries.

---

## Data & Problem

The project focuses on binary fraud detection on tabular transaction data:

- **Input:** Transaction-level features (amount, time, device, geography, etc.).
- **Output:** Probability that the transaction is fraudulent.
- **Challenges:** Strong class imbalance (fraud is rare), need for explainability.

For experimentation, the dataset is split into train/validation/test sets, with all performance numbers reported on a held-out real test set.

---

## Experiments & MLflow

All experiments are **intended** to be driven from Jupyter notebooks and tracked with MLflow. The notebook list and tiers below describe the planned experiment structure; some notebooks or experiments may not exist yet or may be incomplete.

### Core notebooks:
- `01_eda_dataset_overview.ipynb` – Data exploration and leakage checks.
- `02_feature_engineering.ipynb` – Feature variants that later become reusable code under `src/features/`.
- `03_baselines_logreg_xgb.ipynb` – Logistic regression and XGBoost baselines.
- `04_extended_models_*.ipynb` – Extended models (CatBoost, FT-Transformer, GraphSAGE).
- `05_shap_explainability.ipynb` – SHAP analysis for the selected model.

### Model experimentation tiers:

| Tier | Model               | Goal                                                  |
|------|---------------------|-------------------------------------------------------|
| 1    | Logistic Regression | Establish a sanity check baseline.                   |
| 2    | XGBoost             | The standard. Beat this score.                        |
| 3    | CatBoost            | Test if handling categories natively improves score.  |
| 4    | FT-Transformer      | Test if deep learning captures hidden interactions.   |
| 5    | GraphSAGE (GNN)     | Test if "network effects" exist in the data.          |

**Model references:**  
- [FT-Transformer (Revisiting Deep Learning Models for Tabular Data)](https://paperswithcode.com/method/ft-transformer)  
- [GraphSAGE (Inductive Representation Learning on Large Graphs)](http://papers.neurips.cc/paper/6703-inductive-representation-learning-on-large-graphs)

### MLflow logs for each run include:
- **Params:** model type, hyperparameters, feature set, seed, data version.
- **Metrics:** ROC AUC, PR AUC, recall@k, etc.
- **Artifacts:** model artifacts, SHAP plots, confusion matrices.

---

## Application: API and Dashboard

This section describes the **planned** application layer. Once a model is selected from experiments, the goal is to provide:

- **FastAPI service**  
  - Loads the deployed XGBoost model and SHAP explainer.
  - `/predict` endpoint: returns fraud probability, decision, and SHAP feature importance.
  - `/explain` endpoint: uses an LLM (OpenAI GPT) to generate natural language explanations tailored for analysts or customers.

- **Streamlit dashboard**  
  Provides dual-view interface addressing different business needs:
  - **Analyst View**: Technical LLM explanation + SHAP risk drivers with bar charts. Designed for fraud operations teams who need to understand *why* a transaction was flagged to make review decisions and tune rules.
  - **Customer View**: Simple, reassuring LLM-generated message ready for SMS/email. Addresses the communication challenge of explaining fraud blocks to customers without revealing detection logic or creating friction in legitimate transactions.
  - Real-time transaction scoring and model performance summaries.

---

## Synthetic Data Experiments with CTGAN

FraudLM is planned to include experiments on using **CTGAN** ([GitHub](https://github.com/sdv-dev/CTGAN), [paper](https://papers.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan)) to address rarity and privacy:

### CTGAN training
- Train CTGAN on real training data to generate realistic synthetic transaction records.

### Synthetic data quality checks
- Compare marginal distributions and correlations between real and synthetic.
- Inspect how well CTGAN captures the minority (fraud) class.

### Utility experiments (always evaluated on real test data)
Planned experiments will train models under three setups:
- **Real-only training**
- **Synthetic-only training**
- **Real + synthetic (augmentation / oversampling)**

Compare downstream performance on the real test set to quantify:
- How much utility synthetic data provides.
- Whether augmentation helps with rare fraud cases.

These experiments are planned to live primarily in dedicated notebooks (`06_ctgan_synthetic_data.ipynb`, `07_synthetic_vs_real_models.ipynb`) and MLflow experiments, and do not change the core app, which is always trained on real data.

---

## Project Structure

> Note: This is the **intended** layout. Early-stage versions of this repo may not yet include all of these folders/files.

```
FraudLM/
├── data/
│   ├── raw/              # Original transaction data
│   ├── processed/        # Cleaned and feature-ready data
│   └── synthetic/        # CTGAN-generated samples
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── src/
│   ├── config/           # Settings and configuration
│   ├── data/             # Loading, preprocessing, CTGAN utilities
│   ├── features/         # Feature engineering functions
│   ├── models/           # Model implementations (LR, XGBoost, etc.)
│   ├── evaluation/       # Metrics and SHAP logic
│   ├── tracking/         # MLflow helpers
│   ├── api/              # FastAPI application
│   └── ui/               # Streamlit dashboard
├── artifacts/            # Saved models and reports
├── mlruns/               # MLflow experiment storage (local backend)
├── requirements.txt
└── README.md
```

---

## Getting Started

> Note: This section reflects the planned developer experience. In the current early-stage state some commands or paths may need adjustment.

### 1. Setup

This project uses [UV](https://docs.astral.sh/uv/) for fast, reliable Python dependency management.

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env   # Set MLFLOW_TRACKING_URI, OPENAI_API_KEY, data paths, etc.
```

### 2. Run experiments

```bash
# Launch Jupyter
jupyter lab
# Open notebooks in the `notebooks/` folder and run them
```

### 3. Start API and dashboard

```bash
# FastAPI
uvicorn src.api.app:app --reload

# Streamlit
streamlit run src/ui/dashboard.py
```

---

## Roadmap

- [ ] Complete baseline model experiments (Logistic Regression, XGBoost, CatBoost).
- [ ] Add extended models (FT-Transformer, GraphSAGE).
- [ ] Complete CTGAN synthetic data experiments.
- [ ] Extend LLM explanations with prompt optimization and evaluation metrics.
- [ ] Add dashboard monitoring plots and drift checks.
- [ ] Integrate with MLflow Model Registry for promotion of models from "Staging" to "Production".
