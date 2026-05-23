#!/bin/bash
set -e

# ==========================================
# Configuration
# ==========================================
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-asia-southeast1}"
REPO="raptor-eval-repo"
SERVICE_ACCOUNT="raptor-backend@${PROJECT_ID}.iam.gserviceaccount.com"
EVAL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/raptor-eval-worker:latest"

BACKEND_SRC="${RAPTOR_BACKEND_SRC:?Set RAPTOR_BACKEND_SRC to the raptor backend directory}"
EVAL_SRC="$(cd "$(dirname "$0")" && pwd)"

echo "=== RAPTOR Evaluation Cloud Run Deploy ==="
echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo ""

# ==========================================
# Step 0: Ensure Artifact Registry repo exists
# ==========================================
echo "--- Ensuring Artifact Registry repo..."
gcloud artifacts repositories describe ${REPO} \
    --project=${PROJECT_ID} \
    --location=${REGION} 2>/dev/null || \
gcloud artifacts repositories create ${REPO} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --repository-format=docker \
    --description="RAPTOR evaluation images"

# Configure docker auth
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null || true

# ==========================================
# Step 1: Stage build context
# ==========================================
echo "--- Staging build context..."
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT

# Copy evaluation code
cp -r "${EVAL_SRC}/evaluation" "${BUILD_DIR}/evaluation"
cp -r "${EVAL_SRC}/ingestion" "${BUILD_DIR}/ingestion"
cp -r "${EVAL_SRC}/benchmarks" "${BUILD_DIR}/benchmarks"
cp -r "${EVAL_SRC}/datasets" "${BUILD_DIR}/datasets"
cp -r "${EVAL_SRC}/workers" "${BUILD_DIR}/workers"
cp "${EVAL_SRC}/requirements-cloudrun.txt" "${BUILD_DIR}/"

# Copy backend dependencies (core, modules)
mkdir -p "${BUILD_DIR}/backend_deps"
cp -r "${BACKEND_SRC}/core" "${BUILD_DIR}/backend_deps/core"
cp -r "${BACKEND_SRC}/modules" "${BUILD_DIR}/backend_deps/modules"

# Copy Dockerfile
cp "${EVAL_SRC}/Dockerfile" "${BUILD_DIR}/"

# ==========================================
# Step 2: Build and push image
# ==========================================
echo "--- Building image..."
docker build --platform linux/amd64 -t "${EVAL_IMAGE}" "${BUILD_DIR}"

echo "--- Pushing image..."
docker push "${EVAL_IMAGE}"

# ==========================================
# Step 3: Create/update Cloud Run Jobs
# ==========================================

# Tree rebuild job (max 3 parallel tasks)
echo "--- Creating tree rebuild job..."
gcloud run jobs create raptor-tree-rebuild \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --image="${EVAL_IMAGE}" \
    --task-timeout=3600s \
    --max-retries=1 \
    --parallelism=3 \
    --tasks=3 \
    --cpu=4 \
    --memory=8Gi \
    --service-account="${SERVICE_ACCOUNT}" \
    --set-env-vars="TASK_TYPE=rebuild_tree,GCS_BUCKET=raptor-assets,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    2>/dev/null || \
gcloud run jobs update raptor-tree-rebuild \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --image="${EVAL_IMAGE}" \
    --task-timeout=3600s \
    --max-retries=1 \
    --parallelism=3 \
    --tasks=3 \
    --cpu=4 \
    --memory=8Gi \
    --service-account="${SERVICE_ACCOUNT}" \
    --set-env-vars="TASK_TYPE=rebuild_tree,GCS_BUCKET=raptor-assets,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"

# Evaluation job (also used for reanchor; trigger overrides --tasks/--parallelism)
echo "--- Creating evaluation job..."
gcloud run jobs create raptor-evaluation \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --image="${EVAL_IMAGE}" \
    --task-timeout=3600s \
    --max-retries=1 \
    --parallelism=2 \
    --tasks=2 \
    --cpu=4 \
    --memory=8Gi \
    --service-account="${SERVICE_ACCOUNT}" \
    --set-env-vars="TASK_TYPE=evaluate,GCS_BUCKET=raptor-assets,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    2>/dev/null || \
gcloud run jobs update raptor-evaluation \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --image="${EVAL_IMAGE}" \
    --task-timeout=3600s \
    --max-retries=1 \
    --parallelism=2 \
    --tasks=2 \
    --cpu=4 \
    --memory=8Gi \
    --service-account="${SERVICE_ACCOUNT}" \
    --set-env-vars="TASK_TYPE=evaluate,GCS_BUCKET=raptor-assets,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"

echo ""
echo "=== Deploy complete ==="
echo "Tree rebuild job: raptor-tree-rebuild"
echo "Evaluation job:   raptor-evaluation"
echo ""
echo "Next: run ./trigger_pipeline.sh to start the pipeline"
