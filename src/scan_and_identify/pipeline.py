"""End-to-end identification: PIL image -> top-K TCGplayer candidates with metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from collector_vision import NeuralEmbedder, rotate_card_180
from PIL import Image

from scan_and_identify.back_rejector import BackRejector
from scan_and_identify.phash import compute_name_phash, hamming_distance
from scan_and_identify.set_index import SetIndex
from scan_and_identify.tcgplayer.store import TCGStore

Confidence = Literal["good", "fair", "poor"]

PHASH_RERANK_WEIGHT = 0.15  # 15% pHash, 85% embedding (matches predecessor system)

# With rotation_invariant=True, skip the 180° embed when the 0° top score is at
# least this — a right-side-up card never scores this high upside down, and the
# second embed roughly doubles per-scan latency. Upside-down cards score well
# below this at 0°, so they still get the 180° pass.
ROTATION_EARLY_EXIT_SCORE = 0.5


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Thresholds for mapping (top-1 score, gap to top-2) -> confidence tier.

    Defaults calibrated against the 172-scan reference eval. Override per-deploy
    via env vars: SCAN_AND_IDENTIFY_CONF_{GOOD,POOR}_{SCORE,GAP}.
    """

    good_score: float = 0.55
    good_gap: float = 0.15
    poor_score: float = 0.45
    poor_gap: float = 0.05

    @classmethod
    def from_env(cls) -> ConfidenceThresholds:
        import os

        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            return float(raw) if raw else default

        return cls(
            good_score=_f("SCAN_AND_IDENTIFY_CONF_GOOD_SCORE", cls.good_score),
            good_gap=_f("SCAN_AND_IDENTIFY_CONF_GOOD_GAP", cls.good_gap),
            poor_score=_f("SCAN_AND_IDENTIFY_CONF_POOR_SCORE", cls.poor_score),
            poor_gap=_f("SCAN_AND_IDENTIFY_CONF_POOR_GAP", cls.poor_gap),
        )

    def classify(self, top_score: float, gap_to_next: float) -> Confidence:
        if top_score >= self.good_score and gap_to_next >= self.good_gap:
            return "good"
        if top_score < self.poor_score or gap_to_next < self.poor_gap:
            return "poor"
        return "fair"


DEFAULT_THRESHOLDS = ConfidenceThresholds()


def classify_confidence(top_score: float, gap_to_next: float) -> Confidence:
    """Shorthand using default thresholds."""
    return DEFAULT_THRESHOLDS.classify(top_score, gap_to_next)


@dataclass
class Candidate:
    product_id: int
    score: float
    name: str
    set_name: str
    set_abbr: str
    group_id: int
    collector_number: str | None
    rarity: str | None
    image_url: str
    printings: list[str]


@dataclass
class IdentifyResult:
    is_card_back: bool
    candidates: list[Candidate]
    confidence: Confidence | None = None


def _rerank_with_phash(
    hits: list[tuple[float, int]],
    query_phash: np.uint64,
    index: SetIndex,
) -> list[tuple[float, int]]:
    """Reorder (score, pid) hits by 0.85*embedding + 0.15*(1 - hamming/64).

    If `index` lacks a pHash for a candidate, the pHash term degrades to 0
    for that one — embedding-only effectively wins for it. The other
    candidates with hashes still get the rerank benefit.
    """
    reranked: list[tuple[float, int]] = []
    for score, pid in hits:
        ref_phash = index.name_phash_for(pid)
        if ref_phash is None:
            phash_score = 0.0
        else:
            phash_score = 1.0 - hamming_distance(query_phash, ref_phash) / 64.0
        combined = (1.0 - PHASH_RERANK_WEIGHT) * score + PHASH_RERANK_WEIGHT * phash_score
        reranked.append((combined, pid))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked


class IdentifyPipeline:
    def __init__(
        self,
        embedder: NeuralEmbedder,
        index: SetIndex,
        store: TCGStore,
        back_rejector: BackRejector,
        confidence_thresholds: ConfidenceThresholds | None = None,
        rotation_early_exit_score: float = ROTATION_EARLY_EXIT_SCORE,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._store = store
        self._back = back_rejector
        self._thresholds = confidence_thresholds or DEFAULT_THRESHOLDS
        self._rotation_early_exit = rotation_early_exit_score

    def identify(
        self,
        image: Image.Image,
        set_ids: list[int] | None,
        top_k: int,
        rotation_invariant: bool,
    ) -> IdentifyResult:
        # Milo path is unchanged: letterbox to 448×448, embed at 0° (and
        # optionally 180°), pick the better cosine. Orientation chosen here
        # is the master signal — the pHash step below mirrors it.
        rgb = image.convert("RGB")
        img = rgb.resize((448, 448))

        if rotation_invariant:
            emb_a = np.asarray(self._embedder.embed(img), dtype=np.float32)
            hits_a = self._index.search(emb_a, set_ids=set_ids, top_k=top_k)
            if hits_a and hits_a[0][0] >= self._rotation_early_exit:
                # 0° already decisive — the 180° pass can't beat it, skip the
                # second embed (the dominant per-scan cost).
                emb, hits, used_180 = emb_a, hits_a, False
            else:
                rotated = rotate_card_180(img)
                emb_b = np.asarray(self._embedder.embed(rotated), dtype=np.float32)
                hits_b = self._index.search(emb_b, set_ids=set_ids, top_k=top_k)
                if hits_b and (not hits_a or hits_b[0][0] > hits_a[0][0]):
                    emb, hits, used_180 = emb_b, hits_b, True
                else:
                    emb, hits, used_180 = emb_a, hits_a, False
        else:
            emb = np.asarray(self._embedder.embed(img), dtype=np.float32)
            hits = self._index.search(emb, set_ids=set_ids, top_k=top_k)
            used_180 = False

        top_score = hits[0][0] if hits else 0.0
        if self._back.is_back(embedding=emb, top_catalog_score=top_score):
            return IdentifyResult(is_card_back=True, candidates=[], confidence=None)

        # pHash rerank: use the same orientation Milo picked. Skip entirely if
        # the index carries no pHashes (e.g. legacy synthetic catalogs in tests).
        if hits and self._index.name_phash_for(int(hits[0][1])) is not None:
            phash_input = rgb.transpose(Image.ROTATE_180) if used_180 else rgb
            query_phash = compute_name_phash(phash_input)
            hits = _rerank_with_phash(hits, query_phash, self._index)

        candidates: list[Candidate] = []
        for score, product_id in hits:
            p = self._store.product(product_id)
            if p is None:
                continue
            candidates.append(
                Candidate(
                    product_id=product_id,
                    score=float(score),
                    name=p["name"],
                    set_name=p["set_name"] or "",
                    set_abbr=p["set_abbr"] or "",
                    group_id=p["group_id"],
                    collector_number=p["collector_number"],
                    rarity=p["rarity"],
                    image_url=p["image_url"] or "",
                    printings=self._store.printings_for_product(product_id),
                )
            )
        confidence: Confidence | None = None
        if candidates:
            gap = candidates[0].score - (candidates[1].score if len(candidates) > 1 else 0.0)
            confidence = self._thresholds.classify(candidates[0].score, gap)
        return IdentifyResult(is_card_back=False, candidates=candidates, confidence=confidence)
