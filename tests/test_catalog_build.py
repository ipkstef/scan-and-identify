import io
import json

import numpy as np
import pandas as pd
import respx
from httpx import Response
from PIL import Image

from scan_and_identify.catalog_build import build_catalog


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_build_catalog_filters_sealed_and_writes_npz(tmp_path):
    products_path = tmp_path / "products.parquet"
    pd.DataFrame(
        {
            "product_id": [1001, 1002, 9999],
            "group_id": [100, 100, 100],
            "name": ["A", "B", "Sealed"],
            "image_url": [
                f"https://tcgplayer-cdn.tcgplayer.com/product/{p}_200w.jpg"
                for p in [1001, 1002, 9999]
            ],
            "is_sealed": [False, False, True],
        }
    ).to_parquet(products_path)

    out = tmp_path / "catalog.npz"
    cache = tmp_path / "imgs"

    with respx.mock(assert_all_called=False) as m:
        m.get(url__regex=r".*_in_1000x1000\.jpg").mock(return_value=Response(404))
        m.get(url__regex=r".*_200w\.jpg").mock(
            return_value=Response(
                200,
                content=_png((128, 128, 128)),
                headers={"content-type": "image/jpeg"},
            )
        )
        # rate kept high so the test runs fast; concurrency low to keep mock deterministic
        build_catalog(
            products_parquet=products_path,
            out_path=out,
            image_cache_dir=cache,
            rate=100.0,
            concurrency=2,
            batch_size=4,
        )

    data = np.load(out, allow_pickle=False)
    assert data["embeddings"].shape == (2, 128)
    assert sorted(data["card_ids"].tolist()) == ["1001", "1002"]
    spec = json.loads(str(data["embedder_spec"]))
    assert spec["kind"] == "neural"
    assert spec["algo_key"] == "milo1+phash1"
    assert data["name_phashes"].shape == (2,)
    assert data["name_phashes"].dtype == np.uint64


def test_build_catalog_skips_already_cached(tmp_path):
    products_path = tmp_path / "products.parquet"
    pd.DataFrame(
        {
            "product_id": [1001],
            "group_id": [100],
            "name": ["A"],
            "image_url": ["https://tcgplayer-cdn.tcgplayer.com/product/1001_200w.jpg"],
            "is_sealed": [False],
        }
    ).to_parquet(products_path)
    cache = tmp_path / "imgs"
    cache.mkdir()
    (cache / "1001.jpg").write_bytes(_png((50, 50, 50)))

    with respx.mock(assert_all_called=False) as m:
        # No network calls should fire — assert_all_called=False allows zero, but we also
        # verify by not registering any 200 routes.
        m.get(url__regex=r".*").mock(return_value=Response(500))
        out = tmp_path / "catalog.npz"
        build_catalog(
            products_parquet=products_path,
            out_path=out,
            image_cache_dir=cache,
            rate=100.0,
            concurrency=1,
            batch_size=4,
        )

    data = np.load(out, allow_pickle=False)
    assert data["embeddings"].shape == (1, 128)
    assert data["card_ids"].tolist() == ["1001"]


def _make_existing_npz(path, product_ids, *, algo_key="milo1+phash1"):
    """Write a synthetic NPZ in the same shape build_catalog writes. Returns (emb, phash)."""
    rng = np.random.default_rng(42)
    n = len(product_ids)
    emb = rng.standard_normal((n, 128)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    phash = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    np.savez_compressed(
        path,
        embeddings=emb,
        card_ids=np.array([str(pid) for pid in product_ids], dtype="<U36"),
        source="tcgplayer",
        embedder_spec=json.dumps({"kind": "neural", "algo_key": algo_key}),
        built_at="2026-05-19T12:34:56Z",
        name_phashes=phash,
    )
    return emb, phash


def _write_products_parquet(path, product_ids):
    pd.DataFrame(
        {
            "product_id": product_ids,
            "group_id": [100] * len(product_ids),
            "name": [f"Card {pid}" for pid in product_ids],
            "image_url": [
                f"https://tcgplayer-cdn.tcgplayer.com/product/{pid}_200w.jpg"
                for pid in product_ids
            ],
            "is_sealed": [False] * len(product_ids),
        }
    ).to_parquet(path)


def test_build_catalog_reuses_all_rows_when_parquet_unchanged(tmp_path):
    """Same products in old NPZ and new parquet → every row reused, zero HTTP calls."""
    existing = tmp_path / "existing.npz"
    emb_in, phash_in = _make_existing_npz(existing, [1001, 1002, 1003])

    products_path = tmp_path / "products.parquet"
    _write_products_parquet(products_path, [1001, 1002, 1003])
    cache = tmp_path / "imgs"
    out = tmp_path / "catalog.npz"

    with respx.mock(assert_all_called=False) as m:
        # If any HTTP call fires, this 500 would propagate as a failed embed.
        m.get(url__regex=r".*").mock(return_value=Response(500))
        build_catalog(
            products_parquet=products_path,
            out_path=out,
            image_cache_dir=cache,
            reuse_existing=existing,
            rate=100.0,
            concurrency=1,
            batch_size=4,
        )

    data = np.load(out, allow_pickle=False)
    out_ids = [int(c) for c in data["card_ids"].tolist()]
    assert sorted(out_ids) == [1001, 1002, 1003]
    # Each output row matches the synthetic input row for the same pid.
    for i_out, pid in enumerate(out_ids):
        i_in = [1001, 1002, 1003].index(pid)
        np.testing.assert_array_equal(data["embeddings"][i_out], emb_in[i_in])
        assert data["name_phashes"][i_out] == phash_in[i_in]


def test_build_catalog_embeds_only_new_when_reusing(tmp_path):
    """Reuse existing rows for known pids; fetch+embed only the new ones."""
    existing = tmp_path / "existing.npz"
    emb_in, phash_in = _make_existing_npz(existing, [1001, 1002])

    products_path = tmp_path / "products.parquet"
    _write_products_parquet(products_path, [1001, 1002, 1003])
    cache = tmp_path / "imgs"
    out = tmp_path / "catalog.npz"

    with respx.mock(assert_all_called=False) as m:
        # Only the new pid (1003) is allowed to fetch. If 1001 or 1002 hits the network,
        # this would 500 and break the embed.
        m.get(url__regex=r".*/(1001|1002)_.*").mock(return_value=Response(500))
        m.get(url__regex=r".*/1003_in_1000x1000\.jpg").mock(return_value=Response(404))
        m.get(url__regex=r".*/1003_200w\.jpg").mock(
            return_value=Response(
                200, content=_png((50, 50, 50)), headers={"content-type": "image/jpeg"}
            )
        )
        build_catalog(
            products_parquet=products_path,
            out_path=out,
            image_cache_dir=cache,
            reuse_existing=existing,
            rate=100.0,
            concurrency=1,
            batch_size=4,
        )

    data = np.load(out, allow_pickle=False)
    out_ids = [int(c) for c in data["card_ids"].tolist()]
    assert sorted(out_ids) == [1001, 1002, 1003]
    # Reused pids match synthetic input bytes.
    for pid_existing, i_in in [(1001, 0), (1002, 1)]:
        i_out = out_ids.index(pid_existing)
        np.testing.assert_array_equal(data["embeddings"][i_out], emb_in[i_in])
        assert data["name_phashes"][i_out] == phash_in[i_in]
    # New pid has a real unit embedding computed fresh; not equal to either synthetic row.
    i_1003 = out_ids.index(1003)
    new_emb = data["embeddings"][i_1003]
    assert new_emb.shape == (128,)
    assert abs(float(np.linalg.norm(new_emb)) - 1.0) < 1e-5
    for prev in emb_in:
        assert not np.array_equal(new_emb, prev)


def test_build_catalog_full_rebuild_when_algo_key_mismatches(tmp_path):
    """Existing NPZ with a different algo_key → ignore it and rebuild from scratch."""
    existing = tmp_path / "existing.npz"
    _make_existing_npz(existing, [1001, 1002], algo_key="milo0-legacy")

    products_path = tmp_path / "products.parquet"
    _write_products_parquet(products_path, [1001, 1002])
    cache = tmp_path / "imgs"
    out = tmp_path / "catalog.npz"

    with respx.mock(assert_all_called=False) as m:
        # Real HTTP calls MUST happen (full rebuild); register success routes.
        m.get(url__regex=r".*_in_1000x1000\.jpg").mock(return_value=Response(404))
        m.get(url__regex=r".*_200w\.jpg").mock(
            return_value=Response(
                200, content=_png((50, 50, 50)), headers={"content-type": "image/jpeg"}
            )
        )
        build_catalog(
            products_parquet=products_path,
            out_path=out,
            image_cache_dir=cache,
            reuse_existing=existing,
            rate=100.0,
            concurrency=1,
            batch_size=4,
        )

    data = np.load(out, allow_pickle=False)
    existing_data = np.load(existing, allow_pickle=False)
    # Full rebuild → output embeddings are real (computed) and differ from the synthetic ones.
    for i in range(2):
        assert not np.array_equal(data["embeddings"][i], existing_data["embeddings"][i])


def test_build_catalog_reuses_when_out_path_is_the_reuse_source(tmp_path):
    """Round-trip safety: passing the same path as both out and reuse_existing works."""
    catalog = tmp_path / "catalog.npz"
    emb_in, phash_in = _make_existing_npz(catalog, [1001, 1002])

    products_path = tmp_path / "products.parquet"
    _write_products_parquet(products_path, [1001, 1002])
    cache = tmp_path / "imgs"

    with respx.mock(assert_all_called=False) as m:
        m.get(url__regex=r".*").mock(return_value=Response(500))
        build_catalog(
            products_parquet=products_path,
            out_path=catalog,
            image_cache_dir=cache,
            reuse_existing=catalog,
            rate=100.0,
            concurrency=1,
            batch_size=4,
        )

    data = np.load(catalog, allow_pickle=False)
    out_ids = [int(c) for c in data["card_ids"].tolist()]
    assert sorted(out_ids) == [1001, 1002]
    for pid, i_in in [(1001, 0), (1002, 1)]:
        i_out = out_ids.index(pid)
        np.testing.assert_array_equal(data["embeddings"][i_out], emb_in[i_in])
