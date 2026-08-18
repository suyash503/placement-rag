# Deploying

The container serves the React build and the API from one origin, so a deployment
is a single service and there is no CORS to configure.

## What it needs

| Resource | Requirement | Why |
| --- | --- | --- |
| RAM | **~1.5 GB** | PyTorch plus the embedding model and the cross-encoder |
| Disk | ~2 GB image | CPU-only torch is ~200 MB; both models ~180 MB, baked in at build |
| Env | `MONGO_URI`, `GEMINI_API_KEY` | never baked into the image |

The RAM figure rules out most 512 MB free tiers — the container starts, then gets
OOM-killed the moment the reranker loads. Set `RERANK_ENABLED=false` to fit in
roughly 800 MB, at the cost of the ranking quality measured in the README.

## Before deploying

**1. Ingest the data.** The container queries Atlas; it does not populate it. Run
the pipeline once from your machine:

```bash
uv run python backend/scripts/enrich_dataset.py && uv run python backend/scripts/build_indexes.py && uv run python backend/app/rag/ingest.py
```

**2. Create a read-only Atlas user for the deployment.** A public demo only ever
reads. In Atlas → Database Access, add a user with `readAnyDatabase`, and use that
user's connection string in the deployed environment — not the one your ingestion
scripts use.

**3. Open Atlas network access.** Most hosts have no static outbound IP, so
Network Access needs `0.0.0.0/0`. That is safe only in combination with step 2 and
a strong password: the credential becomes the only thing protecting the database.

## Hugging Face Spaces

Free, 16 GB RAM, no card required. The Space needs YAML frontmatter in its
`README.md`, which is why it is deployed from a dedicated `space` branch rather
than from `main`.

Create the Space at [huggingface.co/new-space](https://huggingface.co/new-space) —
SDK **Docker**, hardware **CPU basic**, visibility **Public**.

Add `MONGO_URI` and `GEMINI_API_KEY` under Settings → Variables and secrets, as
**Secrets** (not variables — variables are visible to anyone).

Then push the deploy branch:

```bash
git checkout space && git merge main -m "sync" && git push space space:main
```

Free Spaces sleep after 48 hours of inactivity and take ~40 s to wake.

## Google Cloud Run

Cleaner URL and no sleeping, but needs a billing account even to stay inside the
free tier.

```bash
gcloud run deploy placement-rag --source . --region asia-south1 --memory 2Gi --cpu 2 --allow-unauthenticated --set-env-vars "MONGO_URI=...,GEMINI_API_KEY=..."
```

Prefer Secret Manager over `--set-env-vars` for anything long-lived — env vars are
readable by anyone with console access to the project.

## Any other container host

The image is a plain Dockerfile listening on `$PORT` (default 7860).

```bash
docker build -t placement-rag . && docker run -p 7860:7860 --env-file .env placement-rag
```

## Notes

`/api/chat` is rate limited to 12 questions per IP per 5 minutes. The limiter is
in-process, so it is per-container — a multi-instance deployment would need shared
state such as Redis.
