"""In-memory join layer over the TCGplayer parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _cents_to_dollars(v) -> float | None:
    if v is None or pd.isna(v):
        return None
    return round(float(v) / 100.0, 2)


class TCGStore:
    """Indexed read-only view of the 7 TCGplayer parquet files for a single category."""

    def __init__(
        self,
        products: pd.DataFrame,
        skus: pd.DataFrame,
        groups: pd.DataFrame,
        rarities: dict[int, str],
        printings: dict[int, str],
        conditions: dict[int, str],
        languages: dict[int, str],
        printing_name_to_id: dict[str, int],
        condition_name_to_id: dict[str, int],
        language_name_to_id: dict[str, int],
    ) -> None:
        self._products = products
        self._skus = skus
        self._groups = groups
        self._rarities = rarities
        self._printings = printings
        self._conditions = conditions
        self._languages = languages
        self._printing_name_to_id = printing_name_to_id
        self._condition_name_to_id = condition_name_to_id
        self._language_name_to_id = language_name_to_id
        self._products_by_id = {int(r.product_id): r for r in products.itertuples(index=False)}
        self._groups_by_id = {int(r.group_id): r for r in groups.itertuples(index=False)}
        self._skus_by_product: dict[int, list[Any]] = {}
        for r in skus.itertuples(index=False):
            self._skus_by_product.setdefault(int(r.product_id), []).append(r)

    @classmethod
    def load(cls, parquet_dir: Path) -> TCGStore:
        products = pd.read_parquet(parquet_dir / "products.parquet")
        skus = pd.read_parquet(parquet_dir / "skus.parquet")
        groups = pd.read_parquet(parquet_dir / "groups.parquet")
        rarities_df = pd.read_parquet(parquet_dir / "rarities.parquet")
        printings_df = pd.read_parquet(parquet_dir / "printings.parquet")
        conditions_df = pd.read_parquet(parquet_dir / "conditions.parquet")
        languages_df = pd.read_parquet(parquet_dir / "languages.parquet")

        rarities = dict(zip(rarities_df["rarity_id"], rarities_df["name"]))
        printings = dict(zip(printings_df["printing_id"], printings_df["name"]))
        conditions = dict(zip(conditions_df["condition_id"], conditions_df["name"]))
        languages = dict(zip(languages_df["language_id"], languages_df["name"]))

        return cls(
            products=products,
            skus=skus,
            groups=groups,
            rarities=rarities,
            printings=printings,
            conditions=conditions,
            languages=languages,
            printing_name_to_id={v: k for k, v in printings.items()},
            condition_name_to_id={v: k for k, v in conditions.items()},
            language_name_to_id={v: k for k, v in languages.items()},
        )

    def set_list(self) -> list[dict]:
        df = self._groups.sort_values(by=["is_current", "name"], ascending=[False, True])
        return [
            {
                "group_id": int(r.group_id),
                "name": r.name,
                "abbr": r.abbr,
                "is_current": bool(r.is_current),
            }
            for r in df.itertuples(index=False)
        ]

    def product(self, product_id: int) -> dict | None:
        row = self._products_by_id.get(int(product_id))
        if row is None:
            return None
        group = self._groups_by_id[int(row.group_id)]
        return {
            "product_id": int(row.product_id),
            "name": row.name,
            "clean_name": row.clean_name,
            "group_id": int(row.group_id),
            "set_name": group.name,
            "set_abbr": group.abbr,
            "collector_number": None
            if pd.isna(row.collector_number)
            else str(row.collector_number),
            "rarity": self._rarities.get(row.rarity_id) if not pd.isna(row.rarity_id) else None,
            "is_sealed": bool(row.is_sealed),
            "image_url": row.image_url,
            "tcgplayer_url": row.url,
        }

    def skus_for_product(self, product_id: int) -> list[dict]:
        rows = self._skus_by_product.get(int(product_id), [])
        return [self._sku_to_dict(r) for r in rows]

    def printings_for_product(self, product_id: int) -> list[str]:
        """Return the unique printing names this product is sold in, sorted by
        ``printing_id`` so Normal comes before Foil. Empty list if no SKUs
        exist (sealed product, dropped row, etc)."""
        rows = self._skus_by_product.get(int(product_id), [])
        ids = sorted({int(r.printing_id) for r in rows})
        return [self._printings[pid] for pid in ids if pid in self._printings]

    def search_products(
        self,
        name: str | None = None,
        collector_number: str | None = None,
        set_ids: list[int] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find non-sealed products matching the given filters.

        At least one of ``name`` or ``collector_number`` must be provided —
        callers enforce this; here, passing both as None returns an empty list
        rather than the whole catalog.

        - ``name``: case-insensitive substring match against products.name.
        - ``collector_number``: case-insensitive exact match.
        - ``set_ids``: restricts to the union of these group_ids. Empty list
          rejected (use None for "no lock"). Unknown ids raise KeyError
          (strict, matches /identify's contract — callers should validate
          against /sets before sending).
        """
        if not name and not collector_number:
            return []

        df = self._products
        df = df[df["is_sealed"] == False]  # noqa: E712
        if set_ids is not None:
            if len(set_ids) == 0:
                raise ValueError("set_ids must be None or a non-empty list of group_ids")
            known = set(self._products["group_id"].astype(int).unique().tolist())
            unknown = [int(s) for s in set_ids if int(s) not in known]
            if unknown:
                raise KeyError(f"Unknown set_id {unknown[0]}")
            df = df[df["group_id"].isin([int(s) for s in set_ids])]
        if name:
            df = df[df["name"].str.contains(name, case=False, na=False, regex=False)]
        if collector_number:
            cn = collector_number.strip().lower()
            df = df[df["collector_number"].astype(str).str.lower() == cn]

        df = df.head(int(limit))
        return [self.product(int(pid)) for pid in df["product_id"]]

    def resolve_sku(
        self, product_id: int, printing: str, condition: str, language: str
    ) -> dict | None:
        printing_id = self._printing_name_to_id.get(printing)
        condition_id = self._condition_name_to_id.get(condition)
        language_id = self._language_name_to_id.get(language)
        if None in (printing_id, condition_id, language_id):
            return None
        for r in self._skus_by_product.get(int(product_id), []):
            if (
                int(r.printing_id) == printing_id
                and int(r.condition_id) == condition_id
                and int(r.language_id) == language_id
            ):
                return self._sku_to_dict(r)
        return None

    def _sku_to_dict(self, r) -> dict:
        return {
            "sku_id": int(r.sku_id),
            "product_id": int(r.product_id),
            "printing": self._printings.get(int(r.printing_id)),
            "condition": self._conditions.get(int(r.condition_id)),
            "language": self._languages.get(int(r.language_id)),
            "market_price": _cents_to_dollars(r.market_price_cents),
            "low_price": _cents_to_dollars(r.low_price_cents),
            "mid_price": _cents_to_dollars(r.mid_price_cents),
            "high_price": _cents_to_dollars(r.high_price_cents),
            "direct_low_price": _cents_to_dollars(r.direct_low_price_cents),
        }
