"""Sync TCGplayer parquet files from R2 to a local cache directory."""

from __future__ import annotations

import logging
from datetime import UTC
from pathlib import Path

import boto3

from scan_and_identify.config import R2Config

log = logging.getLogger(__name__)

PARQUET_NAMES = (
    "products.parquet",
    "skus.parquet",
    "groups.parquet",
    "conditions.parquet",
    "printings.parquet",
    "languages.parquet",
    "rarities.parquet",
)


def _client(cfg: R2Config):
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name="auto",
    )


def download_one_parquet(cfg: R2Config, category: int, name: str, dest: Path) -> Path:
    """Download a single parquet from R2 (e.g. ``products.parquet``) to ``dest``.

    Unlike :func:`sync_parquets`, always downloads (no freshness check) and
    writes to an explicit path the caller controls. Used by the
    ``pull-catalog-parquets`` CLI for catalog refresh.
    """
    if name not in PARQUET_NAMES:
        raise ValueError(f"Unknown parquet name {name!r}; expected one of {PARQUET_NAMES}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = _client(cfg)
    key = f"{category}/{name}"
    log.info("Downloading %s -> %s", key, dest)
    s3.download_file(cfg.bucket, key, str(dest))
    return dest


def sync_parquets(cfg: R2Config, category: int, cache_dir: Path) -> dict[str, Path]:
    """Download any parquet whose R2 LastModified is newer than the local mtime.

    Returns a dict mapping each parquet filename to its local path.
    """
    s3 = _client(cfg)
    out_dir = cache_dir / str(category)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for name in PARQUET_NAMES:
        key = f"{category}/{name}"
        local = out_dir / name
        paths[name] = local

        head = s3.head_object(Bucket=cfg.bucket, Key=key)
        remote_mtime = head["LastModified"].astimezone(UTC).timestamp()

        if local.exists() and local.stat().st_mtime >= remote_mtime:
            log.info("Skipping %s (local is fresh)", name)
            continue

        log.info("Downloading %s", name)
        s3.download_file(cfg.bucket, key, str(local))

    return paths
