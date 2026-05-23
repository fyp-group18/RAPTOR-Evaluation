#!/bin/bash
set -e

# ==========================================
# RAPTOR Evaluation Pipeline — Cloud Run
#
# Phase 0: Upload raw datasets to GCS
# Phase 1: Rebuild full_context_aware trees (2 docs, parallel)
# Phase 2: Reanchor eval dataset against fresh trees
# Phase 3: Evaluate all 3 sequences for both docs (parallel)
#
# Fully automated: no manual intervention between phases
# ==========================================
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="asia-southeast1"
GCS_BUCKET="raptor-assets"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AS_AMM_KEY="112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448"
SC10K_KEY="c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0"

# Phase 1 config: rebuild full_context_aware for both docs
TREE_TASKS="${AS_AMM_KEY}:full:AS_AMM_01_000_I1_R1_Feb_2_2018_compressed.pdf|${SC10K_KEY}:full:SC10000AMM_Rev_J.pdf"

# Phase 2 config: reanchor (pipe-delimited manual:cache_key pairs)
CACHE_KEYS="AS-AMM-01-000:${AS_AMM_KEY}|SC10000AMM:${SC10K_KEY}"

# Phase 3 config: evaluate (pipe-delimited cache_key:doc_source:combined_gcs:image_gcs)
EVAL_TASKS="${AS_AMM_KEY}:AS-AMM-01-000:eval-datasets/as_amm_combined.json:eval-datasets/as_amm_image.json|${SC10K_KEY}:SC10000AMM:eval-datasets/sc10k_combined.json:eval-datasets/sc10k_image.json"

echo "=== RAPTOR Evaluation Pipeline (Cloud Run) ==="
echo "Phase 0: Upload datasets"
echo "Phase 1: Rebuild full_context_aware (2 trees)"
echo "Phase 2: Reanchor datasets"
echo "Phase 3: Evaluate (2 docs x 3 sequences)"
echo ""

# ==========================================
# Phase 0: Upload raw datasets to GCS
# ==========================================
echo "--- Phase 0: Uploading raw datasets to GCS..."
gsutil -q cp "${SCRIPT_DIR}/datasets/combined_eval_dataset.json" \
  "gs://${GCS_BUCKET}/eval-datasets/combined_eval_dataset.json"
gsutil -q cp "${SCRIPT_DIR}/datasets/image_eval_dataset.json" \
  "gs://${GCS_BUCKET}/eval-datasets/image_eval_dataset.json"
echo "Phase 0 complete. Datasets uploaded."
echo ""

# ==========================================
# Phase 1: Rebuild full_context_aware trees
# ==========================================
echo "--- Phase 1: Rebuilding full_context_aware trees (2 tasks, parallelism=2)..."
gcloud run jobs execute raptor-tree-rebuild \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --tasks=2 \
  --task-timeout=3600s \
  --update-env-vars="TASK_CONFIGS=${TREE_TASKS}" \
  --wait

echo ""
echo "Phase 1 complete. Both full_context_aware trees rebuilt."
echo ""

# ==========================================
# Phase 2: Reanchor datasets against fresh trees
# ==========================================
echo "--- Phase 2: Reanchoring datasets against fresh full_context_aware trees..."
gcloud run jobs execute raptor-evaluation \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --tasks=1 \
  --task-timeout=1800s \
  --update-env-vars="TASK_TYPE=reanchor,CACHE_KEYS=${CACHE_KEYS},RAW_COMBINED_GCS=eval-datasets/combined_eval_dataset.json,RAW_IMAGE_GCS=eval-datasets/image_eval_dataset.json" \
  --wait

echo ""
echo "Phase 2 complete. Reanchored datasets uploaded to GCS."
echo ""

# ==========================================
# Phase 3: Evaluate all sequences
# ==========================================
echo "--- Phase 3: Running evaluation (2 docs x 3 sequences, parallelism=2)..."
gcloud run jobs execute raptor-evaluation \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --tasks=2 \
  --task-timeout=3600s \
  --update-env-vars="^;^EVAL_TASKS=${EVAL_TASKS};SEQUENCES=1,2,3;TOP_K=1,3,5,10" \
  --wait

echo ""
echo "=== Pipeline complete ==="
echo "Results: gs://${GCS_BUCKET}/eval-results/"
echo ""
echo "Download results:"
echo "  gsutil -m cp -r gs://${GCS_BUCKET}/eval-results/ ./evaluation/results/"
