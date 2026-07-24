"""Rejects card-back scans by comparing to a canonical embedding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scan_and_identify.image_prep import letterbox


class BackRejector:
    def __init__(self, back_embedding: np.ndarray | None) -> None:
        self._back = back_embedding

    @classmethod
    def load(cls, back_image_path: Path | None, embedder) -> BackRejector:
        """Load a 448x448 canonical card back, embed it, and hold the unit vector.

        If the path is None or missing, returns a disabled rejector that never flags.
        """
        if back_image_path is None or not back_image_path.exists():
            return cls(back_embedding=None)
        img = letterbox(Image.open(back_image_path))
        emb = np.asarray(embedder.embed(img), dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        return cls(back_embedding=emb)

    def is_back(self, embedding: np.ndarray, top_catalog_score: float) -> bool:
        if self._back is None:
            return False
        back_score = float(np.dot(embedding, self._back))
        return back_score > top_catalog_score
