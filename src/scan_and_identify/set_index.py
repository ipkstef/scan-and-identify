"""Per-set sub-catalogs for hard set-lock filtering."""

from __future__ import annotations

import numpy as np
from collector_vision import Catalog

from scan_and_identify.tcgplayer.store import TCGStore


class SetIndex:
    """Holds the full catalog plus one Catalog-shaped sub-view per group_id present.

    Optionally carries a parallel ``{product_id -> name_phash}`` lookup so the
    identify pipeline can rerank top-K candidates by perceptual-hash similarity
    against the input image's name region.
    """

    def __init__(
        self,
        full: Catalog,
        by_set: dict[int, Catalog],
        name_phash_by_pid: dict[int, np.uint64] | None = None,
    ) -> None:
        self._full = full
        self._by_set = by_set
        self._name_phash_by_pid = name_phash_by_pid or {}

    @classmethod
    def build(
        cls,
        catalog: Catalog,
        store: TCGStore,
        name_phashes: np.ndarray | None = None,
    ) -> SetIndex:
        product_ids = [int(s) for s in catalog.card_ids]
        products = (store.product(pid) for pid in product_ids)
        group_ids = np.array(
            [p["group_id"] if p else -1 for p in products],
            dtype=np.int64,
        )

        if name_phashes is not None:
            if name_phashes.shape != (len(product_ids),):
                raise ValueError(
                    f"name_phashes shape {name_phashes.shape} doesn't match "
                    f"catalog size ({len(product_ids)})"
                )
            phash_by_pid: dict[int, np.uint64] | None = {
                pid: np.uint64(name_phashes[i]) for i, pid in enumerate(product_ids)
            }
        else:
            phash_by_pid = None

        by_set: dict[int, Catalog] = {}
        for group_id in np.unique(group_ids):
            if group_id == -1:
                continue
            mask = group_ids == group_id
            sub_card_ids = [str(product_ids[i]) for i, m in enumerate(mask) if m]
            sub = Catalog(
                embeddings=catalog.embeddings[mask],
                card_ids=sub_card_ids,
                source=catalog.source,
                embedder_spec=catalog.embedder_spec,
                oracle_ids=None,
            )
            by_set[int(group_id)] = sub
        return cls(full=catalog, by_set=by_set, name_phash_by_pid=phash_by_pid)

    def search(
        self, embedding: np.ndarray, set_ids: list[int] | None, top_k: int
    ) -> list[tuple[float, int]]:
        """Top-K cosine hits, optionally restricted to a union of groups.

        ``set_ids=None`` searches the full catalog. A single-element list takes
        the fast path through that group's pre-sliced sub-catalog. With
        multiple ids, we search each sub-catalog independently and merge —
        cheap for the realistic case (M=2-5). Unknown ids raise ``KeyError``
        (strict): callers should validate before sending.
        """
        if set_ids is None:
            return [
                (score, int(cid))
                for score, cid in self._full.search(embedding, top_k=top_k)
            ]
        if len(set_ids) == 0:
            raise ValueError("set_ids must be None or a non-empty list")

        all_hits: list[tuple[float, int]] = []
        for sid in set_ids:
            if sid not in self._by_set:
                raise KeyError(f"Unknown set_id {sid}")
            target = self._by_set[sid]
            all_hits.extend(
                (score, int(cid)) for score, cid in target.search(embedding, top_k=top_k)
            )
        all_hits.sort(key=lambda x: x[0], reverse=True)
        return all_hits[:top_k]

    def name_phash_for(self, product_id: int) -> np.uint64 | None:
        """Return the catalog's name-region pHash for ``product_id``, or None."""
        return self._name_phash_by_pid.get(int(product_id))
