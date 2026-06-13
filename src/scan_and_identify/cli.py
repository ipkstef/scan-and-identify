"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scan-and-identify")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the FastAPI server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--env-file", default=".env")
    serve.add_argument(
        "--parquet-cache",
        default="data/cache",
        help="Local directory for cached TCGplayer parquets",
    )
    serve.add_argument(
        "--back-image",
        default=None,
        help="Path to a 448x448 canonical card-back PNG (optional)",
    )

    build = sub.add_parser("build-catalog", help="Build a TCGplayer-keyed embedding catalog")
    build.add_argument("--category", type=int, default=1)
    build.add_argument("--products-parquet", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--image-cache", default="data/image_cache")
    build.add_argument(
        "--rate", type=float, default=3.0, help="Max image-CDN requests per second (default 3)"
    )
    build.add_argument(
        "--concurrency", type=int, default=4, help="Max in-flight requests (default 4)"
    )
    build.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Products per download/embed batch (default 256)",
    )
    build.add_argument(
        "--reuse-existing",
        default=None,
        help=(
            "Path to a prior catalog NPZ. Embeddings and pHashes for products "
            "already present in that NPZ are reused verbatim (no fetch, no embed); "
            "only new products are processed. Mismatched algo_key → full rebuild."
        ),
    )

    dl = sub.add_parser(
        "download-images", help="Download all product images into the cache without embedding"
    )
    dl.add_argument("--products-parquet", required=True)
    dl.add_argument("--image-cache", default="data/image_cache")
    dl.add_argument(
        "--rate", type=float, default=10.0, help="Max image-CDN requests per second (default 10)"
    )
    dl.add_argument(
        "--concurrency", type=int, default=16, help="Max in-flight requests (default 16)"
    )
    dl.add_argument(
        "--batch-size", type=int, default=1024, help="Products per download batch (default 1024)"
    )

    pull = sub.add_parser(
        "pull-products-parquet",
        help="Download products.parquet from R2 to a local path (used during catalog refresh)",
    )
    pull.add_argument("--out", required=True, help="Destination path for products.parquet")
    pull.add_argument(
        "--category", type=int, default=1, help="TCGplayer category id (default 1 = MTG)"
    )
    pull.add_argument("--env-file", default=".env")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "serve":
        return _serve(args)
    if args.cmd == "build-catalog":
        return _build_catalog(args)
    if args.cmd == "download-images":
        return _download_images(args)
    if args.cmd == "pull-products-parquet":
        return _pull_products_parquet(args)
    return 2


def _serve(args) -> int:
    load_dotenv(args.env_file)
    import uvicorn

    from scan_and_identify.config import Config
    from scan_and_identify.server.app import create_app
    from scan_and_identify.server.state import AppState

    config = Config.from_env()
    state = AppState.bootstrap(
        config=config,
        parquet_cache=Path(args.parquet_cache),
        back_image=Path(args.back_image) if args.back_image else None,
    )
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _build_catalog(args) -> int:
    from scan_and_identify.catalog_build import build_catalog

    build_catalog(
        products_parquet=Path(args.products_parquet),
        out_path=Path(args.out),
        image_cache_dir=Path(args.image_cache),
        rate=args.rate,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        reuse_existing=Path(args.reuse_existing) if args.reuse_existing else None,
    )
    return 0


def _download_images(args) -> int:
    from scan_and_identify.catalog_build import download_only

    download_only(
        products_parquet=Path(args.products_parquet),
        image_cache_dir=Path(args.image_cache),
        rate=args.rate,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )
    return 0


def _pull_products_parquet(args) -> int:
    load_dotenv(args.env_file)

    from scan_and_identify.config import R2Config
    from scan_and_identify.tcgplayer.r2_sync import download_one_parquet

    cfg = R2Config(
        access_key=os.environ["R2_TCGPLAYER_BUCKET_ACCESS_KEY"],
        secret_key=os.environ["R2_TCGPLAYER_BUCKET_SECRET_KEY"],
        endpoint_url=os.environ["R2_TCGPLAYER_URL"],
    )
    download_one_parquet(cfg, category=args.category, name="products.parquet", dest=Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
