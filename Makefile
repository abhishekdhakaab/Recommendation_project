.PHONY: test \
        download-steam preprocess-steam \
        preprocess-sports \
        train-two-tower train-sasrec train-sasrec-text \
        compute-bge-embeddings compute-bge-embeddings-steam \
        eval-sequence eval-sasrec \
        train-sasrec-steam train-sasrec-text-steam train-two-tower-steam \
        log-sequence-result \
        serve-demo serve-artifact \
        docker-up mlflow-ui

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests serving/tests -q

# ── Data ──────────────────────────────────────────────────────────────────────

# Download Steam dataset files from McAuley Lab (UCSD).
# Files land in data/raw/steam/.  Re-run with --overwrite to force re-download.
download-steam:
	python data/download_steam.py --output-dir data/raw/steam

# Preprocess Steam into the SeqRec pipeline format.
# Requires: download-steam (or files already in data/raw/steam/).
# Output: data/processed/steam/{train,validation,test}.jsonl, item_metadata.jsonl, stats.json
preprocess-steam:
	python data/preprocess_steam.py \
		--reviews  data/raw/steam/australian_user_reviews.json.gz \
		--games    data/raw/steam/steam_games.json.gz \
		--output-dir data/processed/steam

preprocess-sports:
	python data/preprocess_amazon.py \
		--interactions data/raw/Sports_and_Outdoors.jsonl.gz \
		--metadata data/raw/meta_Sports_and_Outdoors.jsonl.gz \
		--output-dir data/processed/sports_and_outdoors

# ── Two-Tower: full 30-epoch run on all 2M interactions ──────────────────────
# Expected outcome: Recall@50 ~0.05–0.12 after convergence.
# Builds HNSW FAISS index automatically.

train-two-tower:
	KMP_DUPLICATE_LIB_OK=TRUE python models/train_two_tower.py \
		--processed-dir data/processed/sports_and_outdoors \
		--output-dir models/artifacts/sports_and_outdoors_tower_full \
		--epochs 30 \
		--batch-size 1024 \
		--patience 5

# ── BGE text embeddings ───────────────────────────────────────────────────────
# Pre-compute BGE-large embeddings for all Sports & Outdoors items.
# Required for (a) cold-start path and (b) text-enriched SASRec.
# Runtime: ~30 min on CPU for 74K items; ~5 min on GPU.

compute-bge-embeddings:
	KMP_DUPLICATE_LIB_OK=TRUE python models/compute_bge_embeddings.py \
		--processed-dir data/processed/sports_and_outdoors \
		--output-path models/artifacts/sports_and_outdoors_bge/bge_item_embeddings.npy \
		--model-path models/artifacts/hf_models/bge-large-en-v1.5 \
		--batch-size 256

# ── SASRec: full 200-epoch run with early stopping ───────────────────────────
# Fixes applied vs original:
#   1. Sliding-window training (~2M examples instead of 319K)
#   2. Popularity-biased negative sampling (harder negatives)
#   3. Sampled evaluation (100 popularity-biased negatives per user, fast)
#   4. Fixed evaluation: index→item_id off-by-one corrected
#   5. Early stopping on validation NDCG@10 (patience=20)
#   6. Cosine LR schedule
# Expected outcome: NDCG@10 ~0.35–0.45, Hit@10 ~0.50–0.65 (sampled eval).

train-sasrec:
	KMP_DUPLICATE_LIB_OK=TRUE python models/train_sasrec.py \
		--processed-dir data/processed/sports_and_outdoors \
		--output-dir models/artifacts/sports_and_outdoors_sasrec_full \
		--epochs 200 \
		--batch-size 256 \
		--negatives-per-positive 100 \
		--n-eval-negatives 100 \
		--max-validation-users 5000 \
		--patience 20

# ── Text-enriched SASRec ─────────────────────────────────────────────────────
# Same as train-sasrec but with BGE item text features added at input.
# Run compute-bge-embeddings first.
# Expected improvement: +3–8% NDCG@10 relative over ID-only SASRec.

train-sasrec-text:
	KMP_DUPLICATE_LIB_OK=TRUE python models/train_sasrec.py \
		--processed-dir data/processed/sports_and_outdoors \
		--output-dir models/artifacts/sports_and_outdoors_sasrec_text \
		--epochs 200 \
		--batch-size 256 \
		--negatives-per-positive 100 \
		--n-eval-negatives 100 \
		--max-validation-users 5000 \
		--patience 20 \
		--text-embeddings-path models/artifacts/sports_and_outdoors_bge/bge_item_embeddings.npy

# ── Evaluation ────────────────────────────────────────────────────────────────

eval-sequence:
	KMP_DUPLICATE_LIB_OK=TRUE python evaluation/evaluate_top_item_sequence_retrieval.py \
		--processed-dir data/processed/sports_and_outdoors \
		--output evaluation/results/sports_and_outdoors/top_item_sequence_5k.json \
		--top-items 5000 \
		--max-users 5000 \
		--n-negatives 100 \
		--embedding-dim 64 \
		--window-size 20 \
		--popularity-weight 0.0

log-sequence-result:
	python evaluation/log_result_to_mlflow.py \
		evaluation/results/sports_and_outdoors/top_item_sequence_5k.json \
		--run-name sports-top5k-sequence \
		--tracking-uri ./mlruns \
		--param dataset=sports_and_outdoors \
		--param protocol=top5k_popularity_sampled

# ── Steam: Two-Tower ─────────────────────────────────────────────────────────
# Requires: preprocess-steam.  ~13K items → faster training than Sports & Outdoors.

train-two-tower-steam:
	KMP_DUPLICATE_LIB_OK=TRUE python models/train_two_tower.py \
		--processed-dir data/processed/steam \
		--output-dir models/artifacts/steam_tower \
		--epochs 30 \
		--batch-size 512 \
		--patience 5

# ── Steam: BGE text embeddings ────────────────────────────────────────────────
# Runtime: ~5 min on CPU for ~13K games.

compute-bge-embeddings-steam:
	KMP_DUPLICATE_LIB_OK=TRUE python models/compute_bge_embeddings.py \
		--processed-dir data/processed/steam \
		--output-path models/artifacts/steam_bge/bge_item_embeddings.npy \
		--model-path models/artifacts/hf_models/bge-large-en-v1.5 \
		--batch-size 256

# ── Steam: SASRec (ID-only) ───────────────────────────────────────────────────
# Expected outcome: NDCG@10 ~0.45–0.55, Hit@10 ~0.65–0.80 (sampled eval, 100 negatives).

train-sasrec-steam:
	KMP_DUPLICATE_LIB_OK=TRUE python models/train_sasrec.py \
		--processed-dir data/processed/steam \
		--output-dir models/artifacts/steam_sasrec \
		--epochs 200 \
		--batch-size 256 \
		--negatives-per-positive 100 \
		--n-eval-negatives 100 \
		--max-validation-users 5000 \
		--patience 20

# ── Steam: Text-enriched SASRec ───────────────────────────────────────────────
# Run compute-bge-embeddings-steam first.
# Expected improvement over ID-only: +3–8% NDCG@10 relative.

train-sasrec-text-steam:
	KMP_DUPLICATE_LIB_OK=TRUE python models/train_sasrec.py \
		--processed-dir data/processed/steam \
		--output-dir models/artifacts/steam_sasrec_text \
		--epochs 200 \
		--batch-size 256 \
		--negatives-per-positive 100 \
		--n-eval-negatives 100 \
		--max-validation-users 5000 \
		--patience 20 \
		--text-embeddings-path models/artifacts/steam_bge/bge_item_embeddings.npy

# ── Serving ───────────────────────────────────────────────────────────────────

serve-demo:
	SEQREC_DEMO_PIPELINE=1 KMP_DUPLICATE_LIB_OK=TRUE \
		uvicorn serving.app.main:app --host 0.0.0.0 --port 8000

serve-artifact:
	SEQREC_ARTIFACT_PIPELINE=1 KMP_DUPLICATE_LIB_OK=TRUE \
		SEQREC_PROCESSED_DIR=data/processed/sports_and_outdoors \
		SEQREC_ARTIFACT_DIR=models/artifacts/sports_and_outdoors_sequence \
		uvicorn serving.app.main:app --host 0.0.0.0 --port 8000

# ── Infrastructure ────────────────────────────────────────────────────────────

docker-up:
	docker compose up --build

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
