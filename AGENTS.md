# SeqRec — Codex Working Instructions

## Project source of truth

The primary specification is `RecSys_Project_PRD.md`.

Codex must follow the PRD unless explicitly instructed otherwise by the user.

Project codename: SeqRec  
Goal: Build a production-style multi-stage recommendation system:
1. Two-tower retrieval
2. FAISS ANN candidate generation
3. SASRec sequential re-scoring
4. BGE/LLM reranking
5. Cold-start handling
6. Redis feature store
7. FastAPI serving
8. Offline evaluation and benchmarking

## Critical working rule

Do not make major design changes without asking first.

Major design changes include:
- Changing the dataset
- Replacing Two-Tower with another retrieval architecture
- Replacing SASRec with another sequential model
- Replacing FAISS with another ANN system
- Replacing Redis with another feature store
- Replacing FastAPI with another serving framework
- Removing MLflow, evaluation, cold-start handling, or monitoring
- Adding new production dependencies not mentioned in the PRD
- Changing repository structure substantially

You may suggest better options, but first explain:
1. What you want to change
2. Why it may be better
3. What tradeoff it creates
4. Whether it affects the PRD goals

Then wait for user approval.

## Development style

Implement the project incrementally.

Do not attempt to build the entire system in one change.

For every task:
1. Inspect the existing repo first.
2. Summarize the files you will change.
3. Make the smallest reasonable implementation.
4. Add tests or verification scripts where appropriate.
5. Run the relevant checks.
6. Summarize what changed and how to verify it.

## Repository structure

Follow the PRD structure unless told otherwise:

seqrec/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── data/
├── models/
├── retrieval/
├── feature_store/
├── serving/
├── evaluation/
├── notebooks/
├── mlflow/
└── monitoring/

## Code quality rules

- Use Python 3.11.
- Prefer simple, readable code over clever abstractions.
- Use type hints for public functions.
- Use Pydantic models for API schemas and feature-store schemas.
- Use PyTorch for neural models.
- Use FAISS for ANN search.
- Use FastAPI for serving.
- Use Redis for online feature serving.
- Use MLflow for experiment tracking where applicable.
- Keep training code separate from serving code.
- Keep offline preprocessing separate from online inference.

## Testing and verification

When modifying code, run the smallest relevant check.

Examples:
- For preprocessing: run on a tiny synthetic sample.
- For model code: run a smoke test forward pass.
- For FAISS code: build a tiny index and query it.
- For FastAPI: run endpoint tests.
- For evaluation: test metrics on known toy examples.
- For Docker changes: validate compose syntax if possible.

Do not claim something works unless you ran a check or clearly state that it was not run.

## Scope guard

Do not add:
- Frontend UI
- GNN models
- distributed training
- PPO/RL recommendations
- full Amazon all-category training
- unnecessary cloud infrastructure
- unnecessary orchestration frameworks

## Definition of done

A task is done only when:
1. The implementation matches the PRD section for that component.
2. Relevant tests or smoke checks pass.
3. The changed files are summarized.
4. Remaining limitations are clearly stated.