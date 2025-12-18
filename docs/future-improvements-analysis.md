# Future Improvements Analysis & Implementation Plan

This document details the technical analysis and implementation strategy for the proposed future improvements to the Fashion MNIST Classifier.

## 1. Model Versioning (MLflow)

### Goal
Track experiments, version models, and manage the model lifecycle.

### Analysis
- **Current**: Models saved as `.keras` files with sidecar `.json` metadata in `local/models` directory.
- **Requirement**: Centralized tracking of parameters (epochs, learning rate), metrics (accuracy, loss), and artifacts (model files).
- **Tool**: [MLflow](https://mlflow.org/).

### Proposed Changes
#### [NEW] `docker-compose.override.yml` (optional) or update `docker-compose.yml`
- Add `mlflow-server` service alongside API and Streamlit.
- Use a local volume/sqlite db for backend store for simplicity initially.

#### [MODIFY] [train.py](src/train.py)
- Import `mlflow`.
- Wrap training logic in `with mlflow.start_run():`.
- Log params: `epochs`, `batch_size`, `model_type`, `learning_rate`.
- Log metrics: `accuracy`, `loss`, `val_accuracy`, `val_loss`.
- Log model: `mlflow.keras.log_model(model, "model")`.

#### [MODIFY] [requirements.txt](requirements.txt)
- Add `mlflow>=2.10.0`.

---

## 2. CI/CD Pipeline (GitHub Actions)

### Goal
Automate testing and deployment/publishing of Docker images.

### Analysis
- **Current**: Manual `pytest` and local docker builds.
- **Requirement**: Run tests on PRs; Build & Push container on merge to `main`.

### Proposed Changes
#### [NEW] `.github/workflows/ci.yml`
- **Triggers**: `push` to `main`, `pull_request`.
- **Job 1: Test**:
    - Checkout code.
    - Set up Python 3.11.
    - Install dependencies.
    - Run `pytest`.
- **Job 2: Build & Push** (Run only on `push` to `main` and if Test passes):
    - Login to GitHub Container Registry (ghcr.io).
    - Build Docker image `fashion-mnist:latest` and `:${{ github.sha }}`.
    - Push images.

---

## 3. Kubernetes Deployment

### Goal
Scalable, production-grade deployment orchestration.

### Analysis
- **Current**: `docker-compose` for local orchestration.
- **Requirement**: Declarative manifests for K8s objects.

### Proposed Changes
#### [NEW] `k8s/namespace.yaml`
- Isolate resources.

#### [NEW] `k8s/api-deployment.yaml`
- `Deployment` with 2 replicas of API.
- `Service` (ClusterIP) for internal load balancing.
- `Liveness/Readiness` probes pointing to `/health` endpoint.

#### [NEW] `k8s/streamlit-deployment.yaml`
- `Deployment` for frontend.
- `Service` (LoadBalancer or NodePort) for external access.
- Env var `API_URL` pointing to the API internal service DNS.

---

## 4. Model Monitoring (Drift Detection)

### Goal
Detect when production data deviates from training data (Data Drift).

### Analysis
- **Current**: No monitoring.
- **Tool**: [Evidently AI](https://github.com/evidentlyai/evidently) (open source, Python-centric).

### Proposed Changes
#### [MODIFY] [main.py](api/main.py)
- Implement a middleware or background task.
- Save a sample of incoming images and their predicted probabilities to a "production" dataset log.

#### [NEW] `src/monitoring.py`
- Script to run periodically (e.g., daily).
- Load "reference" (training) data and "current" (production) data.
- Run `evidently` Report for `DataDriftPreset`.
- Alert (log warning) if drift detected in pixel values or prediction confidence distribution.

---

## 5. A/B Testing Framework

### Goal
Safely test new models against the stable version.

### Analysis
- **Current**: Single model loaded.
- **Requirement**: Route X% of traffic to Model B.

### Proposed Changes
#### [MODIFY] [main.py](api/main.py)
- Load *two* models: `stable_model` (v1) and `canary_model` (v2).
- Add `AB_TEST_RATIO` env var (default 0.0 -> all stable).
- In `/predict`, generate random number (0-1).
- If < Ratio, use Canary; else Stable.
- Add `model_version: "v1" | "v2"` to the response JSON.

#### [MODIFY] [docker-compose.yml]
- Mount both model files.

---

## Execution Priority
1. **CI/CD**: Foundational safety net.
2. **MLflow**: Essential for data science reproducibility.
3. **Monitoring**: Critical for day 2 operations.
4. **Kubernetes**: Scalability (needed when traffic grows).
5. **A/B Testing**: Mature feature for optimization.
