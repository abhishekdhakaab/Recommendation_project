#!/bin/bash
set -e

echo "=== Step 1: Download and preprocess dataset ==="
python data/download_amazon.py --output-dir data/raw/beauty
python data/preprocess.py --input-dir data/raw/beauty --output-dir data/processed/beauty

echo "=== Step 2: Train Two-Tower ==="
KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_two_tower.py \
  --data-dir data/processed/beauty \
  --output-dir models/artifacts/beauty_tower

echo "=== Step 3: Train SASRec ==="
KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_sasrec.py \
  --data-dir data/processed/beauty \
  --output-dir models/artifacts/beauty_sasrec

echo "=== Step 4: Generate BGE Cold-Start Embeddings ==="
KMP_DUPLICATE_LIB_OK=TRUE python models/cold_start.py \
  --meta data/processed/beauty/item_meta.parquet \
  --output-dir models/artifacts/beauty_coldstart

echo "=== Step 5: Run Full Ablation and Evaluation ==="
KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_full_pipeline.py \
  --data-dir data/processed/beauty \
  --artifact-dir models/artifacts

echo "=== Step 6: Populate Redis Feature Store ==="
python data/feature_store_init.py \
  --data-dir data/processed/beauty \
  --artifact-dir models/artifacts

echo "=== Done. Results in evaluation/results/beauty/RESULTS.md ==="
