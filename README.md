# scan-and-identify

HTTP API that identifies Magic cards from scanned images and renders
TCGplayer seller-template CSV exports. Wraps the
[CollectorVision](https://github.com/ipkstef/CollectorVision) engine with a
TCGplayer metadata layer.

See [`docs/API.md`](docs/API.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
and [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## Deploy

### Production: pull the pre-built image

CI publishes images to `ghcr.io/ipkstef/scan-and-identify`. No git clone,
no Docker build on the host.

```bash
# One-time host setup
sudo apt update && sudo apt install -y docker.io docker-compose-plugin

# Pull the docker-compose.yml + .env.example. Two files is all you need.
mkdir scan-and-identify && cd scan-and-identify
curl -fsSLO https://raw.githubusercontent.com/ipkstef/scan-and-identify/main/docker-compose.yml
curl -fsSLO https://raw.githubusercontent.com/ipkstef/scan-and-identify/main/.env.example
cp .env.example .env

# Fill in .env (see "Configuration" below)
nano .env

# Pull and run
docker compose pull && docker compose up -d

# Verify
curl -H "Authorization: Bearer $YOUR_KEY" http://localhost:8000/health
```

Deploy a new image after CI builds one: `docker compose pull && docker compose up -d`.

### Local build

When hacking on the code or not waiting on CI:

```bash
sudo apt install -y docker.io docker-compose-plugin git git-lfs
git lfs install
git clone https://github.com/ipkstef/scan-and-identify.git
cd scan-and-identify
cp .env.example .env
nano .env
docker compose up -d --build   # builds the image locally
```

## Configuration

Required env vars (see `.env.example` for the full list):

| Var | Notes |
|---|---|
| `SCAN_AND_IDENTIFY_API_KEY` | Shared bearer token. Generate with `openssl rand -hex 32`. |
| `SCAN_AND_IDENTIFY_CATALOG_PATH` | Default `/app/catalogs/catalog.npz` (the baked-in catalog). |
| `R2_TCGPLAYER_BUCKET_ACCESS_KEY` | R2 credentials for syncing TCGplayer parquets at boot. |
| `R2_TCGPLAYER_BUCKET_SECRET_KEY` | Same. |
| `R2_TCGPLAYER_URL` | `https://<account-id>.r2.cloudflarestorage.com` |

R2 credentials come from whoever owns the `tcgplayerapi` bucket.

## API

All endpoints require `Authorization: Bearer $SCAN_AND_IDENTIFY_API_KEY`.

| Endpoint | Purpose |
|---|---|
| `GET  /health` | Liveness + catalog version + build timestamp |
| `GET  /sets` | All MTG sets, sorted by name |
| `GET  /search` | Search products by name/collector_number; powers "Find Match" UI |
| `POST /identify` | Single image URL → top-K candidates with confidence tier |
| `POST /identify-batch` | N image URLs → results per image |
| `GET  /products/{id}` | Full product metadata + all SKUs |
| `POST /products/{id}/resolve-sku` | (printing, condition, language) → sku_id + prices |
| `POST /export/tcgplayer-csv` | Confirmed rows → TCGplayer seller-template CSV |

Full reference in [`docs/API.md`](docs/API.md).

## Catalog refresh (monthly)

```bash
scripts/refresh-catalog.sh                            # ~75 min CPU
git add catalogs/catalog.npz && git commit -m "Refresh" && git push
# GitHub Actions → Build image → Run workflow
docker compose pull && docker compose up -d           # on production
```

The refresh runs inside the existing Docker image. Image cache lives at
`~/scan-and-identify-cache/` by default and persists between runs.

## Development

```bash
uv venv -p 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
ruff check src/ tests/
```

Tests use synthetic fixtures — no R2 or real catalog needed.
