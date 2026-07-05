from collector_vision import Catalog, NeuralEmbedder
from PIL import Image

from scan_and_identify.back_rejector import BackRejector
from scan_and_identify.pipeline import IdentifyPipeline
from scan_and_identify.set_index import SetIndex
from scan_and_identify.tcgplayer.store import TCGStore


def _make_pipeline(catalog_path, parquets_path):
    catalog = Catalog.load(catalog_path)
    store = TCGStore.load(parquets_path)
    index = SetIndex.build(catalog, store)
    embedder = NeuralEmbedder()
    return IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        back_rejector=BackRejector(back_embedding=None),
    )


def test_pipeline_returns_candidates_with_metadata(synthetic_catalog, synthetic_parquets):
    pipeline = _make_pipeline(synthetic_catalog, synthetic_parquets)
    img = Image.new("RGB", (448, 448), (200, 50, 50))
    result = pipeline.identify(img, set_ids=None, top_k=3, rotation_invariant=False)
    assert result.is_card_back is False
    assert len(result.candidates) == 3
    top = result.candidates[0]
    assert top.product_id in {1001, 1002, 1003, 2001}
    assert top.name.startswith(("Alpha", "Beta"))
    assert 0.0 <= top.score <= 1.0


def test_pipeline_set_lock_restricts_results(synthetic_catalog, synthetic_parquets):
    pipeline = _make_pipeline(synthetic_catalog, synthetic_parquets)
    img = Image.new("RGB", (448, 448), (200, 50, 50))
    result = pipeline.identify(img, set_ids=[100], top_k=5, rotation_invariant=False)
    assert all(c.group_id == 100 for c in result.candidates)


def test_pipeline_set_ids_union_restricts_to_listed_sets(synthetic_catalog, synthetic_parquets):
    pipeline = _make_pipeline(synthetic_catalog, synthetic_parquets)
    img = Image.new("RGB", (448, 448), (200, 50, 50))
    # Both sets 100 and 200 exist — union should include both
    result = pipeline.identify(img, set_ids=[100, 200], top_k=5, rotation_invariant=False)
    groups = {c.group_id for c in result.candidates}
    assert groups <= {100, 200}
    # Single-set lock should still produce a strict subset of the union
    only_100 = pipeline.identify(img, set_ids=[100], top_k=5, rotation_invariant=False)
    assert all(c.group_id == 100 for c in only_100.candidates)


def test_pipeline_rotation_invariant_does_not_break(synthetic_catalog, synthetic_parquets):
    pipeline = _make_pipeline(synthetic_catalog, synthetic_parquets)
    img = Image.new("RGB", (448, 448), (200, 50, 50))
    result = pipeline.identify(img, set_ids=None, top_k=3, rotation_invariant=True)
    assert len(result.candidates) == 3


class _CountingEmbedder:
    """Delegates to the real embedder, counting embed() calls."""

    def __init__(self):
        self._inner = NeuralEmbedder()
        self.calls = 0

    def embed(self, images):
        self.calls += 1
        return self._inner.embed(images)


def _make_counting_pipeline(catalog_path, parquets_path, early_exit_score):
    catalog = Catalog.load(catalog_path)
    store = TCGStore.load(parquets_path)
    index = SetIndex.build(catalog, store)
    embedder = _CountingEmbedder()
    pipeline = IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        back_rejector=BackRejector(back_embedding=None),
        rotation_early_exit_score=early_exit_score,
    )
    return pipeline, embedder


def test_rotation_early_exit_skips_second_embed_when_score_decisive(
    synthetic_catalog, synthetic_parquets
):
    # Threshold 0.0: any 0° top score qualifies, so the 180° pass must be skipped.
    pipeline, embedder = _make_counting_pipeline(
        synthetic_catalog, synthetic_parquets, early_exit_score=0.0
    )
    img = Image.new("RGB", (448, 448), (200, 50, 50))
    result = pipeline.identify(img, set_ids=None, top_k=3, rotation_invariant=True)
    assert embedder.calls == 1
    assert len(result.candidates) == 3


def test_rotation_still_tries_180_when_score_weak(synthetic_catalog, synthetic_parquets):
    # Threshold above 1.0 can never be met, so both orientations must be embedded.
    pipeline, embedder = _make_counting_pipeline(
        synthetic_catalog, synthetic_parquets, early_exit_score=1.1
    )
    img = Image.new("RGB", (448, 448), (200, 50, 50))
    result = pipeline.identify(img, set_ids=None, top_k=3, rotation_invariant=True)
    assert embedder.calls == 2
    assert len(result.candidates) == 3


def test_pipeline_reranks_by_combined_score(synthetic_catalog, synthetic_parquets):
    """When two candidates are close on embedding but one is much closer on pHash,
    the pHash-similar one wins after rerank.
    """
    import numpy as np

    catalog = Catalog.load(synthetic_catalog)
    store = TCGStore.load(synthetic_parquets)
    embedder = NeuralEmbedder()

    img = Image.new("RGB", (448, 448), (200, 50, 50))
    from scan_and_identify.phash import compute_name_phash

    matching_phash = compute_name_phash(img)
    pids = [int(cid) for cid in catalog.card_ids]

    rng = np.random.default_rng(0)
    phashes = np.array(
        [np.uint64(rng.integers(0, 2**63, dtype=np.uint64)) for _ in pids],
        dtype=np.uint64,
    )

    # Probe embedding-only ordering to identify top-1/top-2 before rerank.
    pre_index = SetIndex.build(catalog, store, name_phashes=None)
    arr = np.asarray(embedder.embed(img.convert("RGB").resize((448, 448))), dtype=np.float32)
    pre_hits = pre_index.search(arr, set_ids=None, top_k=3)
    top1_pid = pre_hits[0][1]
    top2_pid = pre_hits[1][1]

    # Assign hashes so top1 is maximally distant from query, top2 matches perfectly.
    idx_top1 = pids.index(top1_pid)
    idx_top2 = pids.index(top2_pid)
    phashes[idx_top1] = np.uint64(~int(matching_phash) & 0xFFFFFFFFFFFFFFFF)
    phashes[idx_top2] = matching_phash

    index = SetIndex.build(catalog, store, name_phashes=phashes)
    pipeline = IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        back_rejector=BackRejector(back_embedding=None),
    )
    result = pipeline.identify(img, set_ids=None, top_k=3, rotation_invariant=False)
    assert result.candidates[0].product_id == top2_pid


def test_pipeline_skips_rerank_when_phashes_missing(synthetic_catalog, synthetic_parquets):
    """If the SetIndex has no pHashes, the pipeline returns embedding-only ranking unchanged."""
    catalog = Catalog.load(synthetic_catalog)
    store = TCGStore.load(synthetic_parquets)
    embedder = NeuralEmbedder()
    index = SetIndex.build(catalog, store, name_phashes=None)
    pipeline = IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        back_rejector=BackRejector(back_embedding=None),
    )
    img = Image.new("RGB", (448, 448), (200, 50, 50))
    result = pipeline.identify(img, set_ids=None, top_k=3, rotation_invariant=False)
    assert len(result.candidates) == 3
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
