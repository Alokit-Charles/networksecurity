# Network Security Project — Phishing Detection

An end-to-end ML pipeline that ingests phishing/network-security data from MongoDB, validates and transforms it, trains a classification model (with MLflow experiment tracking via DagsHub), and serves training + prediction through a FastAPI app.

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [1. Clone & Install](#1-clone--install)
- [2. Configure Environment Variables](#2-configure-environment-variables)
- [3. Load Data into MongoDB](#3-load-data-into-mongodb)
- [4. Run the Training Pipeline](#4-run-the-training-pipeline)
- [5. Run the FastAPI App](#5-run-the-fastapi-app)
- [6. Run with Docker](#6-run-with-docker)
- [Experiment Tracking (MLflow / DagsHub)](#experiment-tracking-mlflow--dagshub)
- [CI/CD — GitHub Actions](#cicd--github-actions)
- [Troubleshooting](#troubleshooting)

## Architecture

The training pipeline runs as a sequence of stages, each producing an "artifact" consumed by the next stage:

```
MongoDB (raw data)
      │
      ▼
Data Ingestion  ──▶  Data Validation  ──▶  Data Transformation  ──▶  Model Trainer
(export & split)     (schema check,        (KNN-imputation,          (train models,
                       drift report)         preprocessing)            log to MLflow,
                                                                        save model.pkl)
```

- **Data Ingestion** (`networksecurity/components/data_ingestion.py`) — pulls records from MongoDB, writes a feature-store CSV, and splits into train/test sets.
- **Data Validation** (`data_validation.py`) — checks the data against `data_schema/schema.yaml` and generates a drift report.
- **Data Transformation** (`data_transformation.py`) — imputes missing values (KNN imputer) and saves a `preprocessing.pkl` object.
- **Model Trainer** (`model_trainer.py`) — trains candidate models, evaluates them, logs metrics/models to MLflow (tracked on DagsHub), and saves the final `model.pkl`.

All artifacts are written under `Artifacts/<timestamp>/...`, and the final production-ready model + preprocessor are copied to `finalized_models/`.

Serving is handled by `app.py` (FastAPI), which exposes:

| Route      | Method | Purpose                                                        |
|------------|--------|------------------------------------------------------------------|
| `/`        | GET    | Redirects to the interactive API docs (`/docs`)                 |
| `/train`   | GET    | Triggers a full run of `TrainingPipeline`                       |
| `/predict` | POST   | Accepts a CSV upload, runs it through the saved model, returns an HTML table |

## Project Structure

```
networksecurity/
├── .github/workflows/        # CI/CD pipeline (GitHub Actions)
├── Network_Data/              # Source CSV (phisingData.csv)
├── data_schema/schema.yaml    # Expected columns / types for validation
├── networksecurity/
│   ├── components/            # data_ingestion, data_validation, data_transformation, model_trainer
│   ├── constants/training_pipeline/  # all pipeline constants (paths, filenames, thresholds)
│   ├── entity/                 # config_entity.py, artifact_entity.py (dataclasses passed between stages)
│   ├── exception/               # custom NetworkSecurityException
│   ├── logging/                  # custom logger (writes to logs/)
│   ├── pipeline/                 # training_pipeline.py, batch_prediction.py
│   └── utils/                     # main_utils, ml_utils (metrics, model estimator wrapper)
├── templates/table.html       # HTML template used by /predict
├── app.py                     # FastAPI app (train + predict routes)
├── main.py                    # CLI entrypoint that runs the pipeline stage-by-stage
├── push_data.py                # One-off script: load Network_Data CSV into MongoDB
├── test_mongodb.py             # Quick MongoDB connectivity check
├── Dockerfile
├── requirements.txt
└── setup.py
```

## Prerequisites

- Python 3.10+
- A MongoDB Atlas cluster (or any MongoDB instance) and its connection URI
- (Optional, for experiment tracking) A [DagsHub](https://dagshub.com) account connected to this repo
- Docker (optional, for containerized runs)

## 1. Clone & Install

```bash
git clone https://github.com/Alokit-Charles/networksecurity.git
cd networksecurity

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # installs the local `networksecurity` package
```

## 2. Configure Environment Variables

Create a `.env` file in the project root (this is loaded via `python-dotenv`):

```env
MONGO_DB_URL=mongodb+srv://<username>:<password>@<cluster-url>/?appName=<AppName>
```

> **Note:** `test_mongodb.py` currently has a hardcoded URI as an example — replace `<@password>` with your real credentials, or better, update it to read from `.env` like `app.py`/`push_data.py` already do.

If you're using DagsHub for MLflow tracking, also set (or export before running training):

```env
MLFLOW_TRACKING_URI=https://dagshub.com/Alokit-Charles/networksecurity.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

## 3. Load Data into MongoDB

The pipeline reads training data from MongoDB, not directly from the CSV. Push the sample dataset once:

```bash
python push_data.py
```

This converts `Network_Data/phisingData.csv` to JSON records and inserts them into the `NetworkData` collection of the `ALOKITCHARLES` database (see `networksecurity/constants/training_pipeline/__init__.py` to change these names).

Verify connectivity any time with:

```bash
python test_mongodb.py
```

## 4. Run the Training Pipeline

Two equivalent ways:

**A. Directly via CLI**

```bash
python main.py
```

**B. Via the API** (after starting the app — see next section)

```bash
curl http://localhost:8000/train
```

Either way, this runs ingestion → validation → transformation → model training, and writes the final model to `finalized_models/model.pkl` and `finalized_models/preprocessing.pkl`.

## 5. Run the FastAPI App

```bash
python app.py
```

or

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI, where you can:
- Hit `GET /train` to (re)train the model
- Hit `POST /predict` and upload a CSV to get predictions rendered as an HTML table

## 6. Run with Docker

> **Status:** the `Dockerfile` in this repo is a placeholder / not yet finalized, so containerized runs aren't fully set up. Once it's ready:

```bash
docker build -t networksecurity:latest .
docker run --env-file .env -p 8000:8000 networksecurity:latest
```

The app will be available at **http://localhost:8000**.

> Make sure `finalized_models/` (with a trained `model.pkl` + `preprocessing.pkl`) exists before hitting `/predict` — either train locally first and keep the folder, or call `GET /train` once the container is running. Also make sure the app binds to `0.0.0.0` (not `localhost`) inside the container, or it won't be reachable from outside.

## Experiment Tracking (MLflow / DagsHub)

`model_trainer.py` logs `f1_score`, `precision`, `recall`, and the trained model to MLflow, pointed at DagsHub:

```python
dagshub.init(repo_owner='Alokit-Charles', repo_name='networksecurity', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Alokit-Charles/networksecurity.mlflow")
```

View runs at: `https://dagshub.com/Alokit-Charles/networksecurity.mlflow`

## CI/CD — GitHub Actions

`.github/workflows/main.yml` currently defines a single job, **Build & Test**, that runs on every push/PR to `main`:

1. Checks out the repo.
2. Sets up Python 3.10 (with pip caching).
3. Installs `requirements.txt` and the local package (`pip install -e .`), plus `flake8`.
4. Lints with `flake8` — fails the build on real errors (syntax errors, undefined names); style/complexity issues are reported but non-blocking.
5. Does a smoke test by importing the `networksecurity` package and `TrainingPipeline` to catch broken imports early (there's no `tests/` directory yet).

There is **no Docker build/push job yet** — the `Dockerfile` isn't finalized, so a Docker step is intentionally left out for now (rather than committed as a job that would fail). A commented-out template for it sits at the bottom of `main.yml`; once the Dockerfile is ready, uncomment it to add a job that builds the image and pushes it to **GitHub Container Registry (GHCR)**, gated to only run on pushes to `main` after the test job passes.

### Required repository secrets

The current workflow needs **no secrets at all** — it only installs dependencies and runs a lint/import check.

Once you add the Docker job, it uses the automatically-provided `GITHUB_TOKEN` to push to GHCR, so still no extra secrets needed for that step.

If you later want the training pipeline itself to run in CI (e.g. a scheduled retrain job) or want to deploy the image somewhere (AWS/Azure/GCP), add these under **Settings → Secrets and variables → Actions**:

| Secret               | Used for                          |
|----------------------|-------------------------------------|
| `MONGO_DB_URL`        | Connecting to MongoDB during training |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | DagsHub experiment logging |
| *(cloud-specific)*   | e.g. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `ECR_REPOSITORY` if deploying to AWS |

## Troubleshooting

- **`'TrainingPipeline' object has no attribute 'training_pipeline_config'`** — check `training_pipeline.py`; the constructor must be `__init__`, not `__int__` (an existing typo in a couple of files, including `push_data.py`'s `NetworkDataExtract.__int__`).
- **`/predict` fails with a missing file error** — you need to run `/train` (or `main.py`) at least once so `finalized_models/` is populated before predicting.
- **Docker container can't be reached** — `app.py`'s `if __name__ == "__main__"` block binds to `host='localhost'`, which isn't reachable from outside the container. When the `Dockerfile` is finalized, launch with `uvicorn app:app --host 0.0.0.0 --port 8000` instead of `python app.py` to avoid this.
