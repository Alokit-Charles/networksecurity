# Network Security Project — Phishing URL Detection

An end-to-end ML pipeline that ingests phishing/network-security data from MongoDB, validates and transforms it, trains a classification model (with MLflow experiment tracking via DagsHub), syncs artifacts to S3, and serves training + prediction through a FastAPI app — deployed via a self-hosted GitHub Actions runner on EC2.

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [1. Clone & Install](#1-clone--install)
- [2. Configure Environment Variables](#2-configure-environment-variables)
- [3. Load Data into MongoDB](#3-load-data-into-mongodb)
- [4. Run the Training Pipeline](#4-run-the-training-pipeline)
- [5. Run the FastAPI App](#5-run-the-fastapi-app)
- [6. Run with Docker (local)](#6-run-with-docker-local)
- [Experiment Tracking (MLflow / DagsHub)](#experiment-tracking-mlflow--dagshub)
- [S3 Artifact Sync](#s3-artifact-sync)
- [CI/CD — GitHub Actions](#cicd--github-actions)
- [Deploying to EC2 from Scratch](#deploying-to-ec2-from-scratch)
- [Troubleshooting](#troubleshooting)
- [Suggested Improvements](#suggested-improvements)

## Architecture

```
MongoDB (raw data)
      │
      ▼
Data Ingestion  ──▶  Data Validation  ──▶  Data Transformation  ──▶  Model Trainer
(export & split)     (schema check,        (KNN-imputation,          (train 6 candidate models,
                       drift report)         preprocessing.pkl)        log best to MLflow,
                                                                        save model.pkl,
                                                                        sync Artifacts/ + finalized_models/ to S3)
```

- **Data Ingestion** (`networksecurity/components/data_ingestion.py`) — pulls records from MongoDB, writes a feature-store CSV, splits into train/test.
- **Data Validation** (`data_validation.py`) — checks data against `data_schema/schema.yaml`, generates a drift report.
- **Data Transformation** (`data_transformation.py`) — KNN-imputes missing values, saves `preprocessing.pkl` to both the artifact dir and `finalized_models/`.
- **Model Trainer** (`model_trainer.py`) — trains 6 candidate models (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, AdaBoost) with grid search, logs metrics + the best model to MLflow (via DagsHub), saves `finalized_models/model.pkl`.
- **S3 Sync** (`networksecurity/cloud/s3_syncer.py`) — after training, `TrainingPipeline.run_pipeline()` shells out to the AWS CLI to sync `Artifacts/<timestamp>/` and `finalized_models/` to `s3://networksecurity-alokit/...`. This is best-effort: since it runs via `os.system()`, a missing/misconfigured AWS CLI won't crash the pipeline, but the sync will silently no-op.

Serving is handled by `app.py` (FastAPI):

| Route      | Method | Purpose                                                          |
|------------|--------|-------------------------------------------------------------------|
| `/`        | GET    | Redirects to interactive API docs (`/docs`)                       |
| `/train`   | GET    | Triggers a full run of `TrainingPipeline`                         |
| `/predict` | POST   | Accepts a CSV upload, runs it through the saved model, returns an HTML table |

## Project Structure

```
networksecurity/
├── .github/workflows/main.yml   # 3-job CI/CD: test → build & push to ECR → deploy to EC2
├── Network_Data/                # Source CSV (phisingData.csv)
├── data_schema/schema.yaml      # Expected columns/types for validation
├── networksecurity/
│   ├── cloud/s3_syncer.py       # `aws s3 sync` wrapper
│   ├── components/              # data_ingestion, data_validation, data_transformation, model_trainer
│   ├── constants/training_pipeline/  # all pipeline constants (paths, filenames, bucket name, thresholds)
│   ├── entity/                  # config_entity.py, artifact_entity.py
│   ├── exception/                # custom NetworkSecurityException
│   ├── logging/                   # custom logger (writes to logs/<timestamp>.log)
│   ├── pipeline/                   # training_pipeline.py, batch_prediction.py
│   └── utils/                       # main_utils, ml_utils (metrics, model estimator wrapper)
├── templates/table.html         # HTML template used by /predict
├── app.py                       # FastAPI app (train + predict routes)
├── main.py                      # CLI entrypoint, runs the pipeline stage-by-stage
├── push_data.py                 # One-off script: load Network_Data CSV into MongoDB
├── test_mongodb.py              # Quick MongoDB connectivity check
├── Dockerfile
├── requirements.txt
└── setup.py
```

## Prerequisites

- Python 3.10+
- A MongoDB Atlas cluster (or any MongoDB instance) and its connection URI
- An AWS account with:
  - An **ECR repository** to hold the Docker image
  - An **S3 bucket** named `networksecurity-alokit` (or update `TRAINING_BUCKET_NAME` in `networksecurity/constants/training_pipeline/__init__.py` to match your own)
  - An IAM user/role with permissions for **both** ECR (push/pull) and S3 (`s3:PutObject`, `s3:GetObject`, `s3:ListBucket` on the bucket above) — a common gotcha is granting only ECR permissions and having `/train` silently fail to sync
- A [DagsHub](https://dagshub.com) account connected to this repo (for MLflow tracking)
- Docker, for local containerized runs or to mirror what CI does
- An EC2 instance, if you want to replicate the live deployment (see [Deploying to EC2 from Scratch](#deploying-to-ec2-from-scratch))

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

Create a `.env` file in the project root (loaded via `python-dotenv` in `app.py`):

```env
MONGO_DB_URL=mongodb+srv://<username>:<password>@<cluster-url>/?appName=<AppName>
```

> **Note:** `test_mongodb.py` has a hardcoded sample URI — replace it with your real credentials, or better, update it to read from `.env` the way `app.py`/`push_data.py` already do.

If you plan to train locally and want MLflow logging + S3 sync to actually work, also set up:

```bash
# DagsHub — see "Setting up your own DagsHub repo" below for the full walkthrough
export DAGSHUB_USER_TOKEN=<your-dagshub-token>
export DAGSHUB_REPO_OWNER=<your-dagshub-username>
export DAGSHUB_REPO_NAME=<your-dagshub-repo-name>

# AWS CLI — needed for the S3 sync step; run once
aws configure
```

## 3. Load Data into MongoDB

The pipeline reads training data from MongoDB, not directly from the CSV. Push the sample dataset once:

```bash
python push_data.py
```

This inserts `Network_Data/phisingData.csv` into the `NetworkData` collection of the `ALOKITCHARLES` database. Change these names in `networksecurity/constants/training_pipeline/__init__.py` if needed.

Verify connectivity any time with:

```bash
python test_mongodb.py
```

## 4. Run the Training Pipeline

**A. Directly via CLI**

```bash
python main.py
```

**B. Via the API** (after starting the app — see next section)

```bash
curl http://localhost:8080/train
```

Either way, this runs ingestion → validation → transformation → model training → S3 sync, and writes the final model to `finalized_models/model.pkl` and `finalized_models/preprocessing.pkl`.

## 5. Run the FastAPI App

```bash
python app.py
```

or

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open **http://localhost:8080/docs** for the Swagger UI, where you can trigger `/train` and `/predict` (upload a CSV; results render as an HTML table).

## 6. Run with Docker (local)

```bash
docker build -t networksecurity:latest .
docker run --env-file .env -p 8080:8080 networksecurity:latest
```

The app will be reachable at **http://localhost:8080**.

> `finalized_models/` must exist with a trained model before `/predict` will work. Either train locally first and mount that folder in (`-v $(pwd)/finalized_models:/app/finalized_models`), or call `GET /train` once the container is running. Without a volume mount, the model is lost the moment the container is removed.

## Experiment Tracking (MLflow / DagsHub)

`model_trainer.py`'s `track_mlflow()` logs `f1_score`, `precision`, `recall`, and the trained model to MLflow, pointed at DagsHub. The target repo is **configurable via environment variables** (defined in `networksecurity/constants/training_pipeline/__init__.py`), so forks and other users track experiments to their own DagsHub account instead of the original author's:

```python
DAGSHUB_REPO_OWNER: str = os.getenv("DAGSHUB_REPO_OWNER", "Alokit-Charles")
DAGSHUB_REPO_NAME: str = os.getenv("DAGSHUB_REPO_NAME", "networksecurity")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "network-security")
```

`dagshub.init()` also looks for a `DAGSHUB_USER_TOKEN` environment variable and, if found, authenticates non-interactively. **Without it, `dagshub.init()` tries an interactive browser OAuth flow**, which hangs indefinitely on any headless environment (CI runners, EC2, Docker). Always set `DAGSHUB_USER_TOKEN` wherever training actually runs.

### Setting up your own DagsHub repo

1. **Create a DagsHub account** at [dagshub.com](https://dagshub.com) (free tier is fine — sign up with email or GitHub).
2. **Create a repository.** From the DagsHub dashboard, click **"Create +" → "New Repository"**.
   - You can either connect your existing GitHub repo (**"Connect a repo"**, which mirrors it and auto-enables MLflow tracking), or create a blank DagsHub-only repo (**"Create Repository"**) and just use it for tracking — the actual training code doesn't need to live there, only the tracking URI needs to point at it.
   - Name it whatever you like; this becomes your `DAGSHUB_REPO_NAME`.
3. **Note your username/org and repo name** — these are your `DAGSHUB_REPO_OWNER` and `DAGSHUB_REPO_NAME` respectively. They appear directly in the repo's URL: `https://dagshub.com/<DAGSHUB_REPO_OWNER>/<DAGSHUB_REPO_NAME>`.
4. **Generate an access token**: click your avatar (top-right) → **Settings → Tokens** → **"Generate New Token"**. Give it a name, no expiry if you want it long-lived, and copy the value immediately (it's shown once).
5. **Set the environment variables** wherever training runs:

   **Locally** (add to `.env`):
   ```env
   DAGSHUB_USER_TOKEN=<your-token>
   DAGSHUB_REPO_OWNER=<your-dagshub-username>
   DAGSHUB_REPO_NAME=<your-repo-name>
   ```

   **In GitHub Actions** — add `DAGSHUB_USER_TOKEN`, `DAGSHUB_REPO_OWNER`, and `DAGSHUB_REPO_NAME` as repository secrets (Settings → Secrets and variables → Actions). The workflow already passes all three into the deployed container.

   **On a manually-run Docker container**:
   ```bash
   docker run --env-file .env ... networksecurity:latest
   ```

6. **Train and confirm.** Run `/train` (or `python main.py`) once, then check `https://dagshub.com/<DAGSHUB_REPO_OWNER>/<DAGSHUB_REPO_NAME>/experiments` — you should see a new MLflow run with the logged `f1_score`, `precision`, `recall`, and the saved model artifact.

If `DAGSHUB_REPO_OWNER`/`DAGSHUB_REPO_NAME` are left unset, training still works but logs to the original author's DagsHub repo (`Alokit-Charles/networksecurity`) by default — which will fail with a permissions error for anyone who isn't a collaborator on that repo.

View runs at: `https://dagshub.com/<DAGSHUB_REPO_OWNER>/<DAGSHUB_REPO_NAME>.mlflow`

## S3 Artifact Sync

At the end of a successful `run_pipeline()`, two folders get synced to S3 via the AWS CLI:

- `Artifacts/<timestamp>/` → `s3://networksecurity-alokit/artifact/<timestamp>/`
- `finalized_models/` → `s3://networksecurity-alokit/finalized_model/<timestamp>/`

This requires the AWS CLI to be installed (it's baked into the `Dockerfile` via `apt-get install awscli`) and valid AWS credentials with S3 permissions to be present in the environment (same `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION` used elsewhere). If the bucket doesn't exist or credentials lack S3 permissions, the sync fails silently — training still completes and `/train` still returns success, since `S3Sync` doesn't check the shell command's exit code.

## CI/CD — GitHub Actions

`.github/workflows/main.yml` defines three sequential jobs:

**1. `build-and-test` (Continuous Integration)** — runs on GitHub-hosted `ubuntu-latest` for every push/PR to `main`:
- Installs dependencies, lints with `flake8` (blocking only on real syntax/undefined-name errors)
- Smoke-tests the package by importing `TrainingPipeline`

**2. `build-and-push-ecr-image` (Continuous Delivery)** — runs on `ubuntu-latest`, only after job 1 passes:
- Authenticates to AWS, logs into ECR
- Builds the Docker image and pushes it as `<registry>/<ECR_REPOSITORY_NAME>:latest`

**3. `Continuous-Deployment`** — runs on a **self-hosted runner** (your EC2 instance itself), only after job 2 passes:
- Authenticates to AWS, logs into ECR
- Pulls the freshly pushed image
- Force-removes any existing `networksecurity` container (`docker rm -f networksecurity || true` — safe whether or not one exists, running or already stopped)
- Starts a new container, bind-mounting `/home/ubuntu/networksecurity/finalized_models` into `/app/finalized_models` so trained model artifacts **persist across redeploys** instead of being wiped every time the container is recreated
- Prunes unused Docker images/containers to reclaim disk space

### Required repository secrets

Set these under **Settings → Secrets and variables → Actions**:

| Secret                    | Used for                                                                 |
|----------------------------|---------------------------------------------------------------------------|
| `AWS_ACCESS_KEY_ID`        | AWS auth (ECR push/pull, S3 sync)                                        |
| `AWS_SECRET_ACCESS_KEY`    | AWS auth                                                                  |
| `AWS_REGION`               | AWS auth / ECR region                                                    |
| `ECR_REPOSITORY_NAME`      | Bare ECR repo name — used in the **build** job to construct the push tag |
| `AWS_ECR_LOGIN_URI`        | **Full** registry + repo path (e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com/networksecurity`) — used in the **deploy** job to pull/run. Must resolve to the same image as `ECR_REPOSITORY_NAME` above; keeping these in sync is a common source of "image not found" errors. |
| `DAGSHUB_USER_TOKEN`       | Non-interactive MLflow/DagsHub auth (avoids the interactive OAuth hang)  |
| `DAGSHUB_REPO_OWNER`       | Your DagsHub username/org — experiments log here instead of the original author's repo |
| `DAGSHUB_REPO_NAME`        | Your DagsHub repo name                                                    |
| `MONGO_DB_URL`             | MongoDB connection string, passed into the running container             |

## Deploying to EC2 from Scratch

To replicate the live deployment on your own EC2 instance:

1. **Launch an EC2 instance** (Ubuntu recommended) with at least 20GB of disk — repeated Docker builds fill up the default 8GB volume quickly (see [Troubleshooting](#troubleshooting)).
2. **Open the security group** to allow inbound TCP on port `8080` (or whatever port you choose) from `0.0.0.0/0`, in addition to port `22` for SSH.
3. **Install Docker** on the instance and add your user to the `docker` group:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io
   sudo usermod -aG docker $USER && newgrp docker
   ```
4. **Register the instance as a self-hosted GitHub Actions runner**: repo → **Settings → Actions → Runners → New self-hosted runner**, and follow GitHub's generated `./config.sh` / `./run.sh` steps. Run it as a service (`sudo ./svc.sh install && sudo ./svc.sh start`) so it survives reboots and SSH disconnects.
5. **Create the persistent model folder** referenced by the workflow's volume mount:
   ```bash
   mkdir -p /home/ubuntu/networksecurity/finalized_models
   ```
6. **Add all the secrets** listed above to the GitHub repo.
7. **Push to `main`** — the workflow will build, push to ECR, and deploy automatically. On the very first deploy, `finalized_models/` will be empty, so hit `GET /train` once against the running instance to populate it.

## Troubleshooting

Real issues hit while setting this up, and their fixes:

- **CI import step hangs then fails with a `JSONDecodeError`** — `dagshub.init()` was being called at *module import time*, triggering an interactive OAuth flow with no browser available in CI. Fixed by moving `dagshub.init()` inside `track_mlflow()`, and by always setting `DAGSHUB_USER_TOKEN`.
- **`Unable to resolve action... repository not found`** or **`Unrecognized named-value`** in workflow logs — check action names and `secrets.` references for typos; both fail the job, sometimes before any step even runs.
- **`docker build`: `open Dockerfile: no such file or directory`** — the job is missing its own `actions/checkout` step. Every GitHub Actions job runs on a fresh runner/VM; `needs:` only controls ordering, it does **not** share a filesystem between jobs.
- **`apt update` returns `404 Not Found` on `deb.debian.org`** — the base image (`python:3.10-slim-buster`) is EOL and archived. Use `python:3.10-slim-bookworm` (Debian 12) instead.
- **`Could not load credentials from any providers`** on `amazon-ecr-login`** — check that the AWS credential inputs are actually attached to a `configure-aws-credentials` step and not accidentally left dangling under an unrelated step (e.g. `checkout`) after an edit.
- **`docker: Error response from daemon: Conflict. The container name "/networksecurity" is already in use`** — the cleanup step only checked `docker ps` (running containers), missing containers that had already crashed and stopped. Use `docker rm -f networksecurity || true`, which removes the container regardless of its state and doesn't fail the job if it never existed.
- **`/predict` fails with the pickle files missing right after a redeploy** — a fresh container has an empty filesystem; `finalized_models/` isn't part of the repo (it's generated by `/train`) and doesn't survive a container replacement. Mount a host folder via `-v` so the trained model persists across redeploys, and train once after the very first deploy.
- **`pymongo.errors.ServerSelectionTimeoutError: localhost:27017`** — `MONGO_DB_URL` wasn't passed into the container's environment, so `pymongo.MongoClient(None, ...)` silently defaulted to `localhost`. Always pass `-e 'MONGO_DB_URL=...'` explicitly in `docker run`.
- **`pymongo.errors.InvalidURI: ... must begin with 'mongodb://'`** — the secret value itself was malformed (e.g. wrapped in the wrong quote characters when it was set). Re-enter the secret carefully.
- **`No space left on device`, even inside the GitHub Actions runner's own log files** — the EC2 disk filled up from repeated image builds. Run `docker system prune -a -f` (note the `-a`, which also removes non-dangling unused images, not just untagged ones) periodically, and consider adding a larger EBS volume.