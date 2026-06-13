#!/usr/bin/env bash
# Refresh the embedding catalog from the latest TCGplayer parquet.
#
# Runs end-to-end inside the existing scan-and-identify Docker image so the
# operator doesn't need a Python venv or AWS CLI installed locally. Outputs a
# new catalogs/catalog.npz ready to commit.
#
# Usage:
#   scripts/refresh-catalog.sh
#
# Reads .env for R2 credentials. Optional env vars:
#   IMAGE_CACHE=~/scan-and-identify-cache   persistent image cache (default)
#   RATE=40                                 download requests per second
#   CONCURRENCY=32                          in-flight requests
#   IMAGE_TAG=scan-and-identify:latest      docker image to run the build in

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

IMAGE_CACHE="${IMAGE_CACHE:-$HOME/scan-and-identify-cache}"
RATE="${RATE:-40}"
CONCURRENCY="${CONCURRENCY:-32}"
IMAGE_TAG="${IMAGE_TAG:-scan-and-identify:latest}"

if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example and fill in R2 credentials." >&2
    exit 1
fi

mkdir -p "$IMAGE_CACHE"
WORK=$(mktemp -d -t cv-refresh.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

# Make sure we have the image we're going to run the build in.
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo "==> Image $IMAGE_TAG not found locally. Building..."
    docker build -t "$IMAGE_TAG" .
fi

echo "==> [1/3] Pulling latest products.parquet from R2"
docker run --rm \
    --env-file .env \
    -v "$WORK:/work" \
    "$IMAGE_TAG" \
    scan-and-identify pull-products-parquet --out /work/products.parquet

echo "==> [2/3] Building catalog (image cache: $IMAGE_CACHE)"
echo "    Rate: $RATE req/s, concurrency: $CONCURRENCY"
echo "    Reusing prior embeddings from catalogs/catalog.npz if compatible;"
echo "    only new products are fetched + embedded (~75 min embed if all-new)."
docker run --rm \
    --env-file .env \
    -v "$WORK/products.parquet:/work/products.parquet:ro" \
    -v "$IMAGE_CACHE:/work/cache" \
    -v "$(pwd)/catalogs:/work/out" \
    "$IMAGE_TAG" \
    scan-and-identify build-catalog \
        --products-parquet /work/products.parquet \
        --image-cache /work/cache \
        --out /work/out/catalog.npz \
        --reuse-existing /work/out/catalog.npz \
        --rate "$RATE" --concurrency "$CONCURRENCY"

echo "==> [3/3] Done"
ls -lh catalogs/catalog.npz
echo
echo "Next steps:"
echo "    git add catalogs/catalog.npz"
echo "    git commit -m 'Refresh catalog'"
echo "    git push"
echo "    # Then trigger the 'Build image' workflow in GitHub Actions"
echo "    # Then on production: docker compose pull && docker compose up -d"
