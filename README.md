# ♟️ chessMania — Chess MLOps Pipeline

> From raw PGN ingestion to Transformer-powered next-move prediction, with full MLOps lifecycle coverage.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

**chessMania** is an end-to-end MLOps project that demonstrates every stage of the machine learning lifecycle using chess game data. It starts with a **tabular ML baseline (XGBoost)** and scales into **Transformer-based sequence modelling**, showcasing research engineering rigor and modern ML operations practices.

| Component | Implementation |
|---|---|
| **Storage** | MinIO (PGN / JSON / Parquet) |
| **Orchestration** | Apache Airflow (PGN parsing → Feature & Sequence extraction) |
| **Data Validation** | Great Expectations |
| **Data Warehouse** | PostgreSQL |
| **ML Framework** | XGBoost → GPT-style Transformer with LoRA/QLoRA |
| **Experiment Tracking** | MLflow (Accuracy, F1, AUC, Perplexity, MFU) |
| **Serving** | FastAPI + Uvicorn (Next-move prediction, Win probability) |
| **Monitoring** | Evidently AI (Feature drift + Structural/Move-sequence drift) |
| **Containerisation** | Docker + Docker Compose |

---

## 🏗️ Project Structure

```
chessMania/
│
├── README.md                          # ← You are here
├── pyproject.toml                     # Dependencies & build config
├── Makefile                           # Developer shortcuts
├── Dockerfile                         # API container image
├── .dockerignore
├── .gitignore
├── .env.example                       # Environment variable template
├── .pre-commit-config.yaml            # Linting & formatting hooks
├── alembic.ini                        # Database migration config
├── LICENSE
│
├── src/                               # ═══ APPLICATION CODE ═══
│   ├── __init__.py
│   │
│   ├── config/                        # Centralised configuration
│   │   ├── __init__.py                #   Hydra/OmegaConf loader
│   │   └── config.yaml               #   All settings (storage, models, serving…)
│   │
│   ├── ingestion/                     # ── Stage 1: Data Ingestion ──
│   │   ├── __init__.py
│   │   ├── minio_client.py            #   MinIO upload / download helpers
│   │   ├── pgn_parser.py             #   Parse PGN → structured dicts
│   │   ├── db.py                      #   SQLAlchemy models (PostgreSQL)
│   │   ├── ingest_pgn.py             #   End-to-end ingestion entry-point
│   │   └── validate.py               #   Great Expectations quality checks
│   │
│   ├── preprocessing/                 # ── Stage 2: Preprocessing ──
│   │   ├── __init__.py
│   │   ├── tabular_features.py       #   XGBoost feature extraction
│   │   ├── sequence_tokenizer.py     #   SAN → integer token IDs
│   │   ├── dataset.py                #   PyTorch Dataset / DataLoader
│   │   └── splits.py                 #   Train / Val / Test splitting
│   │
│   ├── models/                        # ── Stage 3: Model Development ──
│   │   ├── __init__.py
│   │   ├── mlflow_utils.py           #   MLflow tracking helpers
│   │   ├── xgboost_trainer.py        #   XGBoost classifier training
│   │   ├── transformer_model.py      #   GPT-style causal LM architecture
│   │   ├── transformer_trainer.py    #   Training loop with LoRA/QLoRA
│   │   └── registry.py              #   Save / load model artefacts
│   │
│   ├── serving/                       # ── Stage 4: Deployment & Serving ──
│   │   ├── __init__.py
│   │   ├── app.py                    #   FastAPI application & routes
│   │   ├── schemas.py                #   Pydantic request / response models
│   │   └── inference.py              #   Inference helpers
│   │
│   └── monitoring/                    # ── Stage 5: Model Monitoring ──
│       ├── __init__.py
│       ├── generate_report.py        #   Evidently drift report generator
│       └── drift_detectors.py        #   Custom sequence-level drift checks
│
├── airflow/                           # ═══ ORCHESTRATION ═══
│   └── dags/
│       ├── chess_ingestion_dag.py     #   Daily: ingest → validate → features
│       └── chess_training_dag.py      #   Manual: train models → monitor
│
├── infra/                             # ═══ INFRASTRUCTURE ═══
│   └── docker-compose.yml             #   MinIO, PostgreSQL, MLflow, Airflow, API
│
├── tests/                             # ═══ TEST SUITE ═══
│   ├── __init__.py
│   ├── test_pgn_parser.py
│   ├── test_tokenizer.py
│   ├── test_features.py
│   ├── test_transformer.py
│   ├── test_api.py
│   └── test_drift.py
│
├── notebooks/                         # ═══ EXPLORATION ═══
│   └── .gitkeep
│
├── data/                              # ═══ DATA (git-ignored) ═══
│   ├── raw/                           #   Raw PGN files
│   ├── interim/                       #   Intermediate artefacts
│   └── processed/                     #   Model-ready features & splits
│
├── artefacts/                         # ═══ MODEL ARTEFACTS (git-ignored) ═══
│   ├── models/                        #   Trained model checkpoints
│   └── tokenizers/                    #   Fitted tokenizer JSON
│
└── reports/                           # ═══ MONITORING REPORTS (git-ignored) ═══
    └── .gitkeep
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/nabin2004/chessMania.git
cd chessMania

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install with dev extras
make dev
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Start Infrastructure

```bash
make infra-up    # Starts MinIO, PostgreSQL, MLflow, Airflow
```

| Service | URL |
|---|---|
| MinIO Console | http://localhost:9001 |
| PostgreSQL | localhost:5432 |
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 |
| API | http://localhost:8000 |

### 4. Ingest Data

Place Lichess PGN files in `data/raw/`, then:

```bash
make ingest      # Parse PGNs → PostgreSQL + MinIO
make validate    # Run Great Expectations checks
```

### 5. Train Models

```bash
make train-xgb          # XGBoost baseline
make train-transformer  # Transformer sequence model
```

### 6. Serve

```bash
make serve   # Start FastAPI at localhost:8000
```

### 7. Monitor

```bash
make monitor  # Generate Evidently drift reports
```

---

## 🔌 API Endpoints

### `GET /health`
```json
{ "status": "ok", "models_loaded": ["xgboost", "transformer"] }
```

### `POST /predict/win`
Predict game outcome probabilities from tabular features.
```json
{
  "white_elo": 1500,
  "black_elo": 1450,
  "time_control": "300+3",
  "eco": "B20",
  "moves_played": 10
}
```
→ `{ "white_win": 0.52, "draw": 0.28, "black_win": 0.20, "predicted_result": "1-0" }`

### `POST /predict/next-move`
Suggest next moves from a partial game sequence.
```json
{
  "moves": ["e4", "e5", "Nf3"],
  "num_suggestions": 3,
  "temperature": 1.0
}
```
→ `{ "suggestions": [{"move": "Nc6", "probability": 0.35}, ...] }`

---

## 🧪 Testing

```bash
make test    # pytest with coverage
make lint    # ruff + mypy
```

---

## 📊 MLOps Lifecycle Mapping

```
┌────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                │
│  Lichess PGNs ──► MinIO ──► Airflow ETL ──► PostgreSQL            │
│                              │                                    │
│                    Great Expectations (validation)                 │
└────────────────────────────┬───────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│                     PREPROCESSING                                 │
│  Tabular Features (ELO, ECO, TC)    │   Sequence Tokenizer (SAN)  │
│  ──► XGBoost feature matrix         │   ──► Integer token IDs     │
└────────────────────────────┬───────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│                   MODEL DEVELOPMENT                               │
│  XGBoost Classifier                 │   GPT-style Transformer     │
│  (Accuracy, F1, AUC)               │   (Perplexity, MFU, Acc)    │
│                                     │   + LoRA / QLoRA adapters   │
│                   MLflow Tracking                                 │
└────────────────────────────┬───────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│                      SERVING                                      │
│  FastAPI + Uvicorn + Docker                                       │
│  /predict/win  (XGBoost)    │   /predict/next-move  (Transformer) │
└────────────────────────────┬───────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│                     MONITORING                                    │
│  Evidently AI                                                     │
│  • Tabular drift (ELO distributions, accuracy)                    │
│  • Sequence drift (invalid tokens, structural anomalies)          │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).
