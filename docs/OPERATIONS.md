# Operations

Deploy, refresh, and observe.

## Environment variables

| Var | Required | Default | Notes |
|---|---|---|---|
| `SCAN_AND_IDENTIFY_API_KEY` | yes | — | Shared bearer token. Generate via e.g. `openssl rand -base64 48`. |
| `SCAN_AND_IDENTIFY_CATALOG_PATH` | yes | — | Absolute path to the `.npz` catalog. In Docker, baked in at `/app/catalogs/<file>.npz`. |
| `R2_TCGPLAYER_BUCKET_ACCESS_KEY` | yes | — | R2 access key for the `tcgplayerapi` bucket. |
| `R2_TCGPLAYER_BUCKET_SECRET_KEY` | yes | — | R2 secret key. |
| `R2_TCGPLAYER_URL` | yes | — | R2 endpoint, e.g. `https://<account-id>.r2.cloudflarestorage.com`. |
| `SCAN_AND_IDENTIFY_TCGPLAYER_CATEGORY` | no | `1` | TCGplayer category id. `1` = MTG. |
| `SCAN_AND_IDENTIFY_CONF_GOOD_SCORE` | no | `0.55` | Confidence tier threshold (top-1 score). |
| `SCAN_AND_IDENTIFY_CONF_GOOD_GAP` | no | `0.15` | Confidence tier threshold (gap to top-2). |
| `SCAN_AND_IDENTIFY_CONF_POOR_SCORE` | no | `0.45` | Below this score → `poor`. |
| `SCAN_AND_IDENTIFY_CONF_POOR_GAP` | no | `0.05` | Below this gap → `poor`. |
| `AWS_DEFAULT_REGION` | no | `auto` | Set by boto3 for R2; usually not needed. |

The container fails fast at boot with `ConfigError` if any required var is unset.

## Deployment

Images are built by the **`Build image` GitHub Actions workflow** (manual
trigger) and published to `ghcr.io/ipkstef/scan-and-identify` with two
tags per build:

- `ghcr.io/ipkstef/scan-and-identify:<short-sha>` — pinned, for rollback
- `ghcr.io/ipkstef/scan-and-identify:latest` — convenience

On any production host:

```bash
# One-time setup
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
mkdir scan-and-identify && cd scan-and-identify
curl -fsSLO https://raw.githubusercontent.com/ipkstef/scan-and-identify/main/docker-compose.yml
curl -fsSLO https://raw.githubusercontent.com/ipkstef/scan-and-identify/main/.env.example
cp .env.example .env && nano .env   # fill in secrets

# Deploy or upgrade
docker compose pull && docker compose up -d
```

Rollback to a specific image SHA: edit `docker-compose.yml` to pin
`ghcr.io/ipkstef/scan-and-identify:<sha>` and `docker compose up -d`.

### Resource sizing

| Resource | Need | Why |
|---|---|---|
| CPU | 4-8 cores | Milo embedder pegs ~4 cores; need headroom for parallel requests |
| RAM | 1-2 GB | Catalog (~55 MB) + parquets (~150 MB) + Python/onnxruntime overhead |
| Disk | 500 MB image + ~150 MB R2 cache | Image is heaviest from baked-in catalog + onnxruntime |
| Network | minimal | Image-fetch latency is bounded by consumer storage, not by us |

Single container handles ~10-20 RPS depending on rotation-invariance and
image-fetch latency. Scale horizontally with N replicas behind a
round-robin load balancer; no sticky sessions needed.

## Catalog refresh (monthly)

```bash
# On any machine with Docker, the repo cloned (LFS), and a populated .env:
cd scan-and-identify
./scripts/refresh-catalog.sh

# Then:
git add catalogs/catalog.npz
git commit -m "Refresh catalog $(date +%Y-%m)"
git push

# In GitHub: Actions → Build image → Run workflow (3-5 min).
# On production:
docker compose pull && docker compose up -d
```

The script runs inside the existing Docker image. It pulls the latest
products, skus, and conditions parquets from R2 (skus + conditions decide
which products are sealed), downloads any new images (existing cache
reused), re-embeds the full catalog + computes name-region pHashes
(~75 min CPU, the slow step), and writes `catalogs/catalog.npz`.

Image cache lives at `~/scan-and-identify-cache/` by default; override
with `IMAGE_CACHE=/path/to/cache`. Other knobs: `RATE`, `CONCURRENCY`,
`IMAGE_TAG`.

The image cache persists between runs, so an interrupted refresh resumes
where it stopped (downloads only — embed always re-runs the full set).

### Catalog versioning

The NPZ embeds an `algo_key` field. Current servers require
`milo1+phash1`; older `milo1` catalogs are rejected at boot. If you roll
back the server image to a pre-pHash version, roll the catalog back too.

## Nightly restart for fresh prices

Parquets are a boot-time snapshot; TCGStore doesn't auto-refresh.
TCGplayer publishes new prices roughly daily, so a long-running container
serves stale prices. Restart on cron:

```bash
sudo crontab -e
# Add:
0 3 * * * docker restart scan-and-identify >/dev/null 2>&1
```

Downtime is ~5-10 seconds per restart. A `POST /admin/refresh` endpoint
would let you skip the restart; not built yet.

## Observability

### What to monitor

| Signal | Where | Alert if |
|---|---|---|
| Container alive | Health check (Docker, k8s liveness) | `/health` returns non-200 |
| Boot succeeded | First `/health` reply | Takes >30s (something is wrong) |
| Catalog age | `/health` `catalog_version` field | Doesn't include the current YYYY-MM after the 5th of the month |
| Parquet sync age | `/health` `parquet_synced_at` field | Older than container restart cadence + 1 day |
| Identify latency | App logs (uvicorn access) | p95 > 2s sustained — image fetch is probably slow |
| Identify errors | App logs | Any 5xx; non-trivial rate of 400 |
| 401s | App logs | Burst of 401s = key rotation needed or website misconfigured |
| Memory | Container metrics | RSS climbing > 2 GB (we shouldn't allocate over time) |

### Useful curl commands

```bash
# Liveness
curl -fsS -H "Authorization: Bearer $KEY" http://localhost:8000/health

# Sample identify with a known-good TCGplayer image
curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"image_url":"https://tcgplayer-cdn.tcgplayer.com/product/218276_in_1000x1000.jpg","top_k":3}' \
  http://localhost:8000/identify | jq

# Set list (first 3)
curl -s -H "Authorization: Bearer $KEY" http://localhost:8000/sets | jq '.sets[:3]'
```

`scripts/smoke_test.sh` runs all of these for you.

### Logs

Plain text on stdout, uvicorn's default format. `docker logs -f
scan-and-identify` to tail. Per-identify scoring telemetry lives on lines
prefixed `identify scan_id=`, greppable for tier mix / score distribution
analysis.

## Rotating the API key

Generate a new key, update the website's secret, then redeploy with the
new value in `.env`. Single shared key in v1; no zero-downtime rotation.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Container exits on boot with `ConfigError` | Missing env var | Check `.env`; see required vars table above |
| Container exits with `FileNotFoundError: catalog.npz` | `SCAN_AND_IDENTIFY_CATALOG_PATH` points at a non-existent file | Verify path; rebuild image if catalog was supposed to be baked in |
| Boot hangs >30s | R2 sync is timing out | Check `R2_TCGPLAYER_URL`, network reachability, credentials |
| All `/identify` requests return 400 "Could not fetch image" | Consumer storage is unreachable from the container | Network/firewall; pre-signed URLs may have expired |
| `/identify` returns wrong card consistently for one set | Catalog is stale for that set | Refresh catalog (see above) |
| Slow `/identify` (>2s) | Image fetch from consumer storage is slow | Investigate storage latency; pre-warm with `--rotation_invariant false` for testing |
| 429/503 from TCGplayer CDN during catalog build | Rare; we observed zero in production builds | Drop `--rate` and `--concurrency` in `download-images`; back-rejector pauses 5s automatically |
| OOM during boot | Insufficient container memory | Bump RAM limit to 2 GB |
| Container exits with `ValueError: ... requires 'milo1+phash1'` | Loaded catalog was built before pHash rerank was added | Rebuild with `scripts/refresh-catalog.sh` and redeploy; intentional — no silent fallback |

---

