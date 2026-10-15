"""Unit tests for the catalog-build primitives.

These exercise the primitives independently from the orchestrators
(:func:`download_only`, :func:`build_catalog`), which are covered separately
by tests/test_catalog_build.py.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scan_and_identify.catalog_build import (
    ImageCache,
    RateLimiter,
    embed_one,
    preprocess,
    products_to_fetch,
    read_catalog_built_at,
    tcgplayer_image_urls,
    write_catalog_npz,
)


def _png(size=(300, 300), color=(123, 45, 67)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


# --- tcgplayer_image_urls --------------------------------------------------


def test_urls_high_res_first():
    urls = tcgplayer_image_urls(
        218276, "https://tcgplayer-cdn.tcgplayer.com/product/218276_200w.jpg"
    )
    assert urls[0].endswith("218276_in_1000x1000.jpg")
    assert urls[1].endswith("218276_200w.jpg")


def test_urls_no_fallback_when_parquet_url_missing():
    urls = tcgplayer_image_urls(218276, None)
    assert len(urls) == 1
    assert urls[0].endswith("218276_in_1000x1000.jpg")


def test_urls_dedupe_when_parquet_matches_high_res():
    # If the parquet happens to already be the high-res variant, don't duplicate it.
    high_res = "https://tcgplayer-cdn.tcgplayer.com/product/218276_in_1000x1000.jpg"
    urls = tcgplayer_image_urls(218276, high_res)
    assert urls == [high_res]


# --- products_to_fetch -----------------------------------------------------


def test_products_to_fetch_filters_sealed_and_missing_urls(tmp_path):
    pd.DataFrame(
        {
            "product_id": [1001, 1002, 9999, 1003],
            "name": ["A", "B", "Sealed", "NoURL"],
            "image_url": [
                "https://tcgplayer-cdn.tcgplayer.com/product/1001_200w.jpg",
                "https://tcgplayer-cdn.tcgplayer.com/product/1002_200w.jpg",
                "https://tcgplayer-cdn.tcgplayer.com/product/9999_200w.jpg",
                None,
            ],
        }
    ).to_parquet(tmp_path / "products.parquet")
    # 9999 is sealed: its only SKU is Unopened. 1002 has no SKUs at all → not sealed.
    pd.DataFrame({"product_id": [1001, 1001, 9999, 1003], "condition_id": [1, 2, 6, 1]}).to_parquet(
        tmp_path / "skus.parquet"
    )
    pd.DataFrame(
        {"condition_id": [1, 2, 6], "name": ["Near Mint", "Lightly Played", "Unopened"]}
    ).to_parquet(tmp_path / "conditions.parquet")

    out = list(products_to_fetch(tmp_path))
    pids = [pid for pid, _ in out]
    assert pids == [1001, 1002]  # sealed and null-URL both excluded


def test_products_to_fetch_raises_without_unopened_condition(tmp_path):
    pd.DataFrame({"product_id": [1001], "image_url": ["u"]}).to_parquet(
        tmp_path / "products.parquet"
    )
    pd.DataFrame({"product_id": [1001], "condition_id": [1]}).to_parquet(tmp_path / "skus.parquet")
    pd.DataFrame({"condition_id": [1], "name": ["Near Mint"]}).to_parquet(
        tmp_path / "conditions.parquet"
    )

    with pytest.raises(ValueError, match="Unopened"):
        list(products_to_fetch(tmp_path))


# --- RateLimiter -----------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_paces_acquires():
    import time as _time

    limiter = RateLimiter(rate=20.0)  # 20 req/s -> min interval 50ms

    t0 = _time.monotonic()
    for _ in range(4):
        await limiter.acquire()
    elapsed = _time.monotonic() - t0
    # 4 acquires at 20/s: first is instant, then 3 × 50ms gap = ~150ms total
    assert elapsed >= 0.14  # some tolerance for OS jitter
    assert elapsed < 0.5  # but not absurdly slow


# --- ImageCache ------------------------------------------------------------


def test_image_cache_roundtrip(tmp_path):
    cache = ImageCache(tmp_path / "imgs")
    assert not cache.has(1001)
    assert cache.get(1001) is None

    cache.put(1001, b"hello")
    assert cache.has(1001)
    assert cache.get(1001) == b"hello"
    assert cache.path(1001).name == "1001.jpg"


def test_image_cache_creates_directory(tmp_path):
    cache_dir = tmp_path / "nested" / "imgs"
    ImageCache(cache_dir)  # constructor creates the dir as a side effect
    assert cache_dir.exists()
    assert cache_dir.is_dir()


# --- preprocess ------------------------------------------------------------


def test_preprocess_returns_448x448_rgb():
    img = preprocess(_png(size=(300, 420)))
    assert img.size == (448, 448)
    assert img.mode == "RGB"


def test_preprocess_letterboxes_non_square_with_white():
    # Tall image: 100×400 → scaled to ~112×448 inside 448×448 canvas.
    # Pixels at the left edge of the canvas should be the white background.
    img = preprocess(_png(size=(100, 400), color=(0, 0, 0)))
    px = img.getpixel((0, 224))  # left edge, middle row
    assert px == (255, 255, 255)


def test_preprocess_raises_on_garbage_bytes():
    with pytest.raises(Exception):  # noqa: B017 - PIL.UnidentifiedImageError
        preprocess(b"definitely not an image")


# --- embed_one -------------------------------------------------------------


def test_embed_one_returns_unit_vector():
    from collector_vision import NeuralEmbedder

    embedder = NeuralEmbedder()
    img = preprocess(_png(color=(200, 50, 50)))
    emb = embed_one(embedder, img)
    assert emb.shape == (128,)
    assert emb.dtype == np.float32
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5


# --- write_catalog_npz -----------------------------------------------------


def test_write_catalog_npz_roundtrip(tmp_path):
    import json

    embeddings = np.random.default_rng(0).standard_normal((3, 128)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    ids = ["1001", "1002", "1003"]
    out = tmp_path / "catalog.npz"

    write_catalog_npz(out, ids, embeddings)

    data = np.load(out, allow_pickle=False)
    assert data["embeddings"].shape == (3, 128)
    assert data["card_ids"].tolist() == ids
    assert str(data["source"]) == "tcgplayer"
    assert json.loads(str(data["embedder_spec"]))["algo_key"] == "milo1"


def test_write_catalog_npz_accepts_alternate_source_and_algo(tmp_path):
    import json

    out = tmp_path / "c.npz"
    write_catalog_npz(
        out, ["1"], np.zeros((1, 128), dtype=np.float32), source="custom", algo_key="milo2"
    )
    data = np.load(out, allow_pickle=False)
    assert str(data["source"]) == "custom"
    assert json.loads(str(data["embedder_spec"]))["algo_key"] == "milo2"


def test_write_catalog_npz_bakes_built_at_by_default(tmp_path):
    out = tmp_path / "c.npz"
    write_catalog_npz(out, ["1"], np.zeros((1, 128), dtype=np.float32))
    built_at = read_catalog_built_at(out)
    assert built_at is not None
    # Default is current UTC ISO timestamp
    assert built_at.endswith("Z")
    assert "T" in built_at


def test_write_catalog_npz_accepts_explicit_built_at(tmp_path):
    out = tmp_path / "c.npz"
    write_catalog_npz(
        out, ["1"], np.zeros((1, 128), dtype=np.float32), built_at="2025-01-15T00:00:00Z"
    )
    assert read_catalog_built_at(out) == "2025-01-15T00:00:00Z"


def test_read_catalog_built_at_returns_none_for_old_npz(tmp_path):
    # Simulate an old NPZ that doesn't have the built_at key
    import json

    out = tmp_path / "old.npz"
    np.savez_compressed(
        out,
        embeddings=np.zeros((1, 128), dtype=np.float32),
        card_ids=np.array(["1"], dtype="<U36"),
        source="tcgplayer",
        embedder_spec=json.dumps({"kind": "neural", "algo_key": "milo1"}),
    )
    assert read_catalog_built_at(out) is None


# --- name_phashes ----------------------------------------------------------


def test_write_catalog_npz_writes_name_phashes(tmp_path):
    import json

    out = tmp_path / "catalog.npz"
    embeddings = np.random.default_rng(0).standard_normal((2, 128)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    phashes = np.array([0xDEADBEEF, 0xCAFEF00D], dtype=np.uint64)
    write_catalog_npz(out, ["1", "2"], embeddings, name_phashes=phashes)
    data = np.load(out, allow_pickle=False)
    assert data["name_phashes"].tolist() == [0xDEADBEEF, 0xCAFEF00D]
    assert data["name_phashes"].dtype == np.uint64
    assert json.loads(str(data["embedder_spec"]))["algo_key"] == "milo1+phash1"


def test_write_catalog_npz_without_phashes_keeps_old_algo_key(tmp_path):
    import json

    out = tmp_path / "catalog.npz"
    embeddings = np.random.default_rng(0).standard_normal((1, 128)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    write_catalog_npz(out, ["1"], embeddings)
    data = np.load(out, allow_pickle=False)
    assert "name_phashes" not in data.files
    assert json.loads(str(data["embedder_spec"]))["algo_key"] == "milo1"


def test_write_catalog_npz_rejects_mismatched_phash_length(tmp_path):
    out = tmp_path / "catalog.npz"
    embeddings = np.zeros((2, 128), dtype=np.float32)
    phashes = np.array([0x1, 0x2, 0x3], dtype=np.uint64)
    try:
        write_catalog_npz(out, ["1", "2"], embeddings, name_phashes=phashes)
    except ValueError as e:
        assert "doesn't match" in str(e)
    else:
        raise AssertionError("expected ValueError for length mismatch")


def test_read_catalog_name_phashes_returns_none_when_absent(tmp_path):
    from scan_and_identify.catalog_build import read_catalog_name_phashes

    out = tmp_path / "catalog.npz"
    embeddings = np.zeros((1, 128), dtype=np.float32)
    write_catalog_npz(out, ["1"], embeddings)
    assert read_catalog_name_phashes(out) is None


def test_read_catalog_name_phashes_roundtrip(tmp_path):
    from scan_and_identify.catalog_build import read_catalog_name_phashes

    out = tmp_path / "catalog.npz"
    embeddings = np.zeros((2, 128), dtype=np.float32)
    phashes = np.array([0xAAAA, 0xBBBB], dtype=np.uint64)
    write_catalog_npz(out, ["1", "2"], embeddings, name_phashes=phashes)
    loaded = read_catalog_name_phashes(out)
    assert loaded is not None
    assert loaded.tolist() == [0xAAAA, 0xBBBB]
    assert loaded.dtype == np.uint64
