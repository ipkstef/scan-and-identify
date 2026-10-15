"""Derive which products are sealed from the TCGplayer SKU table.

products.parquet carries no sealed flag. A product is sealed iff it has at
least one SKU in the "Unopened" condition — TCGplayer only sells sealed
product in that condition, and singles never use it.
"""

from __future__ import annotations

import pandas as pd

UNOPENED_CONDITION = "Unopened"


def sealed_product_ids(skus: pd.DataFrame, conditions: pd.DataFrame) -> set[int]:
    """Return the product_ids that have any SKU in the Unopened condition.

    Raises ``ValueError`` if conditions has no Unopened row — without it we
    can't tell sealed from singles, and silently cataloguing sealed product
    is worse than failing.
    """
    unopened = conditions.loc[conditions["name"] == UNOPENED_CONDITION, "condition_id"]
    if unopened.empty:
        raise ValueError(f"conditions.parquet has no {UNOPENED_CONDITION!r} condition")
    mask = skus["condition_id"].isin(unopened.tolist())
    return set(skus.loc[mask, "product_id"].astype(int).tolist())
