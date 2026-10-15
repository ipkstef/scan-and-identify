"""Generate tiny synthetic CollectorVision catalog + TCGplayer parquet files for tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_synthetic_catalog(path: Path, product_ids: list[int]) -> None:
    """Write a CollectorVision-format NPZ with random unit vectors + random pHashes."""
    rng = np.random.default_rng(42)
    n = len(product_ids)
    embeddings = rng.standard_normal((n, 128)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    name_phashes = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    np.savez_compressed(
        path,
        embeddings=embeddings,
        card_ids=np.array([str(pid) for pid in product_ids], dtype="<U36"),
        source="tcgplayer",
        embedder_spec=json.dumps({"kind": "neural", "algo_key": "milo1+phash1"}),
        built_at="2026-05-19T12:34:56Z",
        name_phashes=name_phashes,
    )


def write_synthetic_parquets(out_dir: Path) -> None:
    """Write the 7 TCGplayer parquet files into out_dir, matching the real schema."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "condition_id": [1, 2, 3, 4, 5, 6],
            "name": [
                "Near Mint",
                "Lightly Played",
                "Moderately Played",
                "Heavily Played",
                "Damaged",
                "Unopened",
            ],
        }
    ).to_parquet(out_dir / "conditions.parquet")

    pd.DataFrame({"printing_id": [1, 2], "name": ["Normal", "Foil"]}).to_parquet(
        out_dir / "printings.parquet"
    )

    pd.DataFrame(
        {"language_id": [1, 2, 3, 7], "name": ["English", "Chinese (S)", "Chinese (T)", "Japanese"]}
    ).to_parquet(out_dir / "languages.parquet")

    pd.DataFrame(
        {"rarity_id": [1, 2, 3, 4], "name": ["Common", "Uncommon", "Rare", "Mythic Rare"]}
    ).to_parquet(out_dir / "rarities.parquet")

    pd.DataFrame(
        {
            "group_id": [100, 200],
            "name": ["Test Set Alpha", "Test Set Beta"],
            "abbr": ["TSA", "TSB"],
        }
    ).to_parquet(out_dir / "groups.parquet")

    pd.DataFrame(
        {
            "product_id": [1001, 1002, 1003, 2001, 9999],
            "group_id": [100, 100, 100, 200, 100],
            "name": [
                "Alpha Card 1",
                "Alpha Card 2",
                "Alpha Card 3",
                "Beta Card 1",
                "Alpha Sealed Box",
            ],
            "clean_name": [
                "Alpha Card 1",
                "Alpha Card 2",
                "Alpha Card 3",
                "Beta Card 1",
                "Alpha Sealed Box",
            ],
            "image_url": [
                f"https://tcgplayer-cdn.tcgplayer.com/product/{pid}_200w.jpg"
                for pid in [1001, 1002, 1003, 2001, 9999]
            ],
            "url": [
                f"https://www.tcgplayer.com/product/{pid}/"
                for pid in [1001, 1002, 1003, 2001, 9999]
            ],
            "rarity_id": [1, 2, 3, 1, None],
            "collector_number": ["1", "2", "3", "1", None],
            "subtype": [None, None, None, None, None],
        }
    ).to_parquet(out_dir / "products.parquet")

    skus = []
    sku_id = 5000
    for pid in [1001, 1002, 1003, 2001]:
        for printing_id in [1, 2]:
            for condition_id in [1, 2]:
                skus.append(
                    {
                        "sku_id": sku_id,
                        "product_id": pid,
                        "language_id": 1,
                        "printing_id": printing_id,
                        "condition_id": condition_id,
                        "low_price_cents": 100.0 + sku_id % 50,
                        "mid_price_cents": 150.0 + sku_id % 50,
                        "high_price_cents": 500.0 + sku_id % 50,
                        "market_price_cents": 200.0 + sku_id % 50,
                        "direct_low_price_cents": None if sku_id % 3 == 0 else 90.0,
                    }
                )
                sku_id += 1
    # Sealed product: sold only in the Unopened condition, which is what marks it sealed.
    skus.append(
        {
            "sku_id": sku_id,
            "product_id": 9999,
            "language_id": 1,
            "printing_id": 1,
            "condition_id": 6,
            "low_price_cents": 10000.0,
            "mid_price_cents": 11000.0,
            "high_price_cents": 15000.0,
            "market_price_cents": 12000.0,
            "direct_low_price_cents": None,
        }
    )
    pd.DataFrame(skus).to_parquet(out_dir / "skus.parquet")
