"""TCGplayer catalog build pipeline — primitives + orchestrators.

The file is organized as **primitives** (top half) followed by **orchestrators**
(bottom half). Each primitive does one thing and is independently testable;
the orchestrators are thin compositions.

Primitives
----------
- :func:`products_to_fetch` — read products.parquet → iter[(product_id, urls)]
- :class:`RateLimiter`      — async token bucket (shared across N workers)
- :func:`download_image`    — try a list of URLs, return bytes of first success
- :class:`ImageCache`       — disk-backed cache keyed by product_id
- :func:`preprocess`        — bytes → 448×448 PIL Image (letterboxed)
- :func:`embed_one`         — embedder + PIL Image → (128,) L2-normalised vector
- :func:`write_catalog_npz` — write the NPZ format CollectorVision's Catalog expects

Orchestrators
-------------
- :func:`download_only`     — populate the ImageCache for every product in a parquet
- :func:`build_catalog`     — full pipeline: cache or fetch → preprocess → embed → NPZ
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from datetime import UTC
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from collector_vision import NeuralEmbedder
from PIL import Image

from scan_and_identify.phash import compute_name_phash

log = logging.getLogger(__name__)

USER_AGENT = "scan-and-identify/0.1 (catalog-build; +https://github.com/ipkstef/scan-and-identify)"


# =============================================================================
# Primitives
# =============================================================================


def tcgplayer_image_urls(product_id: int, image_url_from_parquet: str | None = None) -> list[str]:
    """Return the URL list to try for a TCGplayer product image, best-quality first.

    Tries the ``_in_1000x1000.jpg`` high-res variant first; falls back to whatever
    URL is in the parquet (typically the ``_200w.jpg`` thumbnail). Knowing the
    TCGplayer CDN URL pattern lives in exactly this one place.
    """
    urls = [f"https://tcgplayer-cdn.tcgplayer.com/product/{int(product_id)}_in_1000x1000.jpg"]
    if image_url_from_parquet and image_url_from_parquet not in urls:
        urls.append(image_url_from_parquet)
    return urls


def products_to_fetch(products_parquet: Path) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(product_id, urls)`` for every non-sealed product with an image URL.

    URLs are ordered best-quality first; consumers should try them in order.
    Sealed products are excluded because we don't catalog them. Products with no
    parquet image URL at all are silently skipped — they have no image to fetch.
    """
    df = pd.read_parquet(products_parquet)
    df = df[df["is_sealed"] == False]  # noqa: E712
    df = df[df["image_url"].notna()]
    for row in df.itertuples(index=False):
        pid = int(row.product_id)
        yield pid, tcgplayer_image_urls(pid, row.image_url)


class RateLimiter:
    """Async token-bucket: ``acquire()`` returns at most once every ``1/rate`` seconds.

    Shared across N workers via a single asyncio.Lock. With N=32 workers and
    rate=10, you get a steady 10 req/s aggregate with up to 32 in flight.
    """

    def __init__(self, rate: float) -> None:
        self._min_interval = 1.0 / rate
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed = max(self._next_allowed, now) + self._min_interval


async def download_image(
    client: httpx.AsyncClient,
    urls: list[str],
    limiter: RateLimiter,
    *,
    timeout: float = 20.0,
) -> bytes | None:
    """Try each URL in order; return bytes of the first 200-with-image response.

    Each request waits its turn on the rate limiter. On 429/503 (rate-limit /
    overload) the entire pipeline pauses 5s before trying the next URL.
    Returns ``None`` if every URL fails — caller decides what to do.
    """
    for url in urls:
        await limiter.acquire()
        try:
            r = await client.get(url, timeout=timeout)
        except httpx.HTTPError as e:
            log.debug("download error for %s: %s", url, e)
            continue
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
            return r.content
        if r.status_code in (429, 503):
            log.warning("Backpressure %s on %s — sleeping 5s", r.status_code, url)
            await asyncio.sleep(5)
    return None


class ImageCache:
    """Disk-backed image cache, keyed by product_id, stored as ``<dir>/<pid>.jpg``.

    Read-through and write-through. Lookup is O(1) (filesystem stat). The cache
    is the unit of incremental progress: pre-cached IDs are skipped on rebuild,
    so monthly catalog refreshes only fetch new products.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def path(self, product_id: int) -> Path:
        return self._root / f"{int(product_id)}.jpg"

    def has(self, product_id: int) -> bool:
        return self.path(product_id).exists()

    def get(self, product_id: int) -> bytes | None:
        p = self.path(product_id)
        return p.read_bytes() if p.exists() else None

    def put(self, product_id: int, content: bytes) -> None:
        self.path(product_id).write_bytes(content)


def preprocess(image_bytes: bytes, size: int = 448) -> Image.Image:
    """Decode JPEG/PNG bytes → 448×448 RGB PIL Image with letterbox padding.

    Letterboxing preserves aspect ratio by adding white bars rather than
    stretching. This matches what the embedder expects — Milo was trained on
    448×448 inputs and ArcFace embeddings degrade when the aspect ratio is
    distorted.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def embed_one(embedder: NeuralEmbedder, img: Image.Image) -> np.ndarray:
    """Embed a single 448×448 PIL Image → (128,) float32 unit vector.

    Wraps the engine's :meth:`NeuralEmbedder.embed` with explicit L2 normalisation
    (defence-in-depth: the engine should already normalise, but cosine search
    silently produces garbage if you ever feed it a non-unit vector).
    """
    emb = np.asarray(embedder.embed(img), dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm > 1e-8:
        emb = emb / norm
    return emb


def write_catalog_npz(
    path: Path,
    card_ids: list[str],
    embeddings: np.ndarray,
    *,
    source: str = "tcgplayer",
    algo_key: str | None = None,
    built_at: str | None = None,
    name_phashes: np.ndarray | None = None,
) -> None:
    """Write an NPZ in the format CollectorVision's :class:`Catalog` reads.

    ``embeddings`` should already be L2-normalised (use :func:`embed_one`).
    ``card_ids`` should be parallel to embeddings — same length, same order.
    ``built_at`` is an ISO-8601 UTC timestamp; defaults to "now".
    ``name_phashes``, if provided, is a parallel uint64 array of 64-bit
    perceptual hashes of the card name region. When phashes are present the
    default ``algo_key`` is bumped to ``"milo1+phash1"`` to fence off catalogs
    that lack the new field — pass an explicit ``algo_key`` to override.
    """
    from datetime import datetime

    path.parent.mkdir(parents=True, exist_ok=True)
    if built_at is None:
        built_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if algo_key is None:
        algo_key = "milo1+phash1" if name_phashes is not None else "milo1"

    arrays: dict[str, np.ndarray] = {
        "embeddings": embeddings,
        "card_ids": np.array(card_ids, dtype="<U36"),
        "source": np.asarray(source),
        "embedder_spec": np.asarray(json.dumps({"kind": "neural", "algo_key": algo_key})),
        "built_at": np.asarray(built_at),
    }
    if name_phashes is not None:
        if name_phashes.dtype != np.uint64:
            name_phashes = name_phashes.astype(np.uint64)
        if name_phashes.shape != (len(card_ids),):
            raise ValueError(
                f"name_phashes length {name_phashes.shape} doesn't match card_ids ({len(card_ids)})"
            )
        arrays["name_phashes"] = name_phashes
    np.savez_compressed(path, **arrays)


def read_catalog_built_at(path: Path) -> str | None:
    """Return the ``built_at`` ISO timestamp from a catalog NPZ, or None if absent.

    Old NPZs from before this field was added return None — callers should
    fall back to the file's mtime or just report "unknown".
    """
    data = np.load(path, allow_pickle=False)
    if "built_at" in data.files:
        return str(data["built_at"])
    return None


def read_catalog_name_phashes(path: Path) -> np.ndarray | None:
    """Return the ``name_phashes`` uint64 array, or None if the NPZ lacks it.

    Old NPZs from before pHash rerank was added return None — callers should
    treat that as "fall back to embedding-only" and/or hard-fail at boot
    depending on policy.
    """
    data = np.load(path, allow_pickle=False)
    if "name_phashes" not in data.files:
        return None
    arr = data["name_phashes"]
    return arr.astype(np.uint64) if arr.dtype != np.uint64 else arr


# =============================================================================
# Internal helpers used only by orchestrators
# =============================================================================


async def _download_pending(
    pending: list[tuple[int, list[str]]],
    cache: ImageCache,
    *,
    rate: float,
    concurrency: int,
) -> dict[int, bytes]:
    """Parallel-fetch every (pid, urls) in `pending` and write successes to `cache`.

    Returns the {pid: bytes} dict for in-process consumers (e.g. orchestrators
    that want to embed immediately without re-reading from disk).
    """
    limiter = RateLimiter(rate)
    sem = asyncio.Semaphore(concurrency)
    results: dict[int, bytes] = {}
    headers = {"User-Agent": USER_AGENT, "Accept": "image/*"}

    async with httpx.AsyncClient(headers=headers, http2=False) as client:

        async def worker(pid: int, urls: list[str]) -> None:
            async with sem:
                content = await download_image(client, urls, limiter)
                if content is not None:
                    cache.put(pid, content)
                    results[pid] = content

        await asyncio.gather(*(worker(pid, urls) for pid, urls in pending))

    return results


def _log_progress(done: int, total: int, start_t: float, suffix: str = "") -> None:
    elapsed = time.monotonic() - start_t
    rate = done / elapsed if elapsed > 0 else 0
    eta_min = ((total - done) / rate) / 60 if rate > 0 else 0
    log.info(
        "Progress %d/%d (%.1f%%) — %.2f products/s — ETA %.1f min%s",
        done,
        total,
        100 * done / total if total else 0,
        rate,
        eta_min,
        suffix,
    )


# =============================================================================
# Orchestrators
# =============================================================================


def download_only(
    products_parquet: Path,
    image_cache_dir: Path,
    *,
    rate: float = 10.0,
    concurrency: int = 16,
    batch_size: int = 1024,
) -> None:
    """Populate the image cache for every product in ``products_parquet``.

    Already-cached IDs are skipped, so this is resumable: kill it and restart
    with the same ``image_cache_dir`` to continue.

    No embedding, no NPZ. Use this first to fully cache images at high rate,
    then run :func:`build_catalog` to embed without network. Useful when you
    want to separate the slow network phase from CPU embedding.
    """
    cache = ImageCache(image_cache_dir)
    items = list(products_to_fetch(products_parquet))
    total = len(items)
    log.info(
        "Downloading images for %d products (rate=%.1f req/s, concurrency=%d, batch=%d)",
        total,
        rate,
        concurrency,
        batch_size,
    )

    start = time.monotonic()
    cached_at_start = 0
    newly_fetched = 0
    for batch_start in range(0, total, batch_size):
        batch = items[batch_start : batch_start + batch_size]
        pending: list[tuple[int, list[str]]] = []
        for pid, urls in batch:
            if cache.has(pid):
                cached_at_start += 1
            else:
                pending.append((pid, urls))

        if pending:
            fetched = asyncio.run(
                _download_pending(pending, cache, rate=rate, concurrency=concurrency)
            )
            newly_fetched += len(fetched)

        _log_progress(
            batch_start + len(batch),
            total,
            start,
            suffix=f" (cached {cached_at_start}, fetched {newly_fetched})",
        )

    log.info(
        "Done. Cached at start: %d, newly fetched: %d, total images in cache: %d",
        cached_at_start,
        newly_fetched,
        cached_at_start + newly_fetched,
    )


def _load_reuse_rows(
    reuse_existing: Path | None,
) -> dict[int, tuple[np.ndarray, np.uint64]]:
    """Load ``{pid: (embedding, name_phash)}`` from a prior catalog NPZ for reuse.

    Returns an empty dict (→ full rebuild) when:
    - ``reuse_existing`` is None,
    - the file does not exist,
    - the embedder_spec ``algo_key`` is not the current ``milo1+phash1``,
    - the NPZ lacks ``name_phashes``.
    Each skip reason is logged so refresh logs are self-explanatory.

    Arrays are eagerly copied so the file handle can be released before the
    caller overwrites ``reuse_existing`` (the refresh script passes the same
    path as both ``out`` and ``reuse_existing``).
    """
    if reuse_existing is None:
        return {}
    if not reuse_existing.exists():
        log.info("Reuse source %s does not exist — full rebuild", reuse_existing)
        return {}

    expected_algo = "milo1+phash1"
    with np.load(reuse_existing, allow_pickle=False) as data:
        try:
            spec = json.loads(str(data["embedder_spec"]))
        except (KeyError, ValueError):
            log.info("Reuse source %s has no/unparseable embedder_spec — full rebuild", reuse_existing)
            return {}
        if spec.get("algo_key") != expected_algo:
            log.info(
                "Reuse source %s algo_key=%r != %r — full rebuild",
                reuse_existing,
                spec.get("algo_key"),
                expected_algo,
            )
            return {}
        if "name_phashes" not in data.files:
            log.info("Reuse source %s lacks name_phashes — full rebuild", reuse_existing)
            return {}
        emb = data["embeddings"].copy()
        ids = data["card_ids"].copy()
        phash = data["name_phashes"].copy()

    return {int(ids[i]): (emb[i], np.uint64(phash[i])) for i in range(len(ids))}


def build_catalog(
    products_parquet: Path,
    out_path: Path,
    image_cache_dir: Path,
    *,
    rate: float = 3.0,
    concurrency: int = 4,
    batch_size: int = 256,
    reuse_existing: Path | None = None,
) -> None:
    """Build a TCGplayer-keyed embedding catalog from ``products_parquet``.

    For each product: use cached image if present, else fetch from CDN (within
    rate limit). Then preprocess → embed → accumulate. At the end, write NPZ.

    Network politeness: at most ``rate`` requests/sec across all workers, cap
    of ``concurrency`` in-flight requests. Default 3 req/s × 4 workers is
    sustainable on Cloudflare-fronted CDNs without abuse detection.

    When ``reuse_existing`` points at a prior catalog NPZ with the same
    ``algo_key``, embeddings and pHashes for products already present in that
    NPZ are reused verbatim; only new products are fetched and embedded. A
    mismatched ``algo_key`` (e.g. after a model bump) is treated as "fall
    through to full rebuild" — the safe behaviour, since stale embeddings
    can't be silently mixed with new ones.
    """
    cache = ImageCache(image_cache_dir)
    items = list(products_to_fetch(products_parquet))
    total = len(items)
    reuse_by_pid = _load_reuse_rows(reuse_existing)
    if reuse_by_pid:
        log.info("Reuse source loaded: %d rows available", len(reuse_by_pid))
    log.info(
        "Building catalog from %d products (rate=%.1f req/s, concurrency=%d, batch=%d)",
        total,
        rate,
        concurrency,
        batch_size,
    )

    embedder: NeuralEmbedder | None = None
    embeddings: list[np.ndarray] = []
    name_phashes: list[np.uint64] = []
    card_ids: list[str] = []
    reused = 0
    embedded = 0

    start = time.monotonic()
    for batch_start in range(0, total, batch_size):
        batch = items[batch_start : batch_start + batch_size]

        # Phase 0: short-circuit pids whose embeddings we already have.
        to_embed: list[tuple[int, list[str]]] = []
        for pid, urls in batch:
            row = reuse_by_pid.get(pid)
            if row is not None:
                emb, phash = row
                embeddings.append(emb)
                name_phashes.append(phash)
                card_ids.append(str(pid))
                reused += 1
            else:
                to_embed.append((pid, urls))

        if to_embed:
            # Phase 1: collect already-cached images, queue the rest for fetch.
            in_memory: dict[int, bytes] = {}
            pending: list[tuple[int, list[str]]] = []
            for pid, urls in to_embed:
                cached = cache.get(pid)
                if cached is not None:
                    in_memory[pid] = cached
                else:
                    pending.append((pid, urls))

            # Phase 2: fetch the missing ones (also writes to cache).
            if pending:
                fetched = asyncio.run(
                    _download_pending(pending, cache, rate=rate, concurrency=concurrency)
                )
                in_memory.update(fetched)

            # Phase 3: preprocess + embed + pHash in batch order. Lazy-init the
            # embedder so a 100%-reuse refresh skips loading the ONNX model.
            if embedder is None:
                embedder = NeuralEmbedder()
            for pid, _ in to_embed:
                content = in_memory.get(pid)
                if content is None:
                    continue
                try:
                    img = preprocess(content)
                    raw = Image.open(io.BytesIO(content)).convert("RGB")
                    phash = compute_name_phash(raw)
                except Exception as e:
                    log.warning("Bad image for product %s: %s", pid, e)
                    continue
                embeddings.append(embed_one(embedder, img))
                name_phashes.append(phash)
                card_ids.append(str(pid))
                embedded += 1

        _log_progress(batch_start + len(batch), total, start)

    log.info("Embedded %d new products; reused %d from prior catalog", embedded, reused)
    write_catalog_npz(
        out_path,
        card_ids,
        np.stack(embeddings, axis=0),
        name_phashes=np.array(name_phashes, dtype=np.uint64),
    )
    log.info("Wrote %d embeddings to %s", len(card_ids), out_path)


# =============================================================================
# Async iteration helper (exported for future streaming consumers)
# =============================================================================


async def stream_image_bytes(
    items: list[tuple[int, list[str]]],
    cache: ImageCache,
    *,
    rate: float,
    concurrency: int,
) -> AsyncIterator[tuple[int, bytes]]:
    """Async-yield ``(product_id, image_bytes)`` for each item, fetching as needed.

    Yields cached-first (no network), then fetches missing in parallel. Useful
    for a streaming embed pipeline that doesn't need to hold all items in
    memory at once.

    Not used by the current orchestrators (they accumulate in-memory because
    embedding is much slower than yielding). Kept here as a composition point
    for future work.
    """
    cached = [(pid, cache.get(pid)) for pid, _ in items]
    missing = [(pid, urls) for pid, urls in items if not cache.has(pid)]

    for pid, content in cached:
        if content is not None:
            yield pid, content

    if not missing:
        return

    fetched = await _download_pending(missing, cache, rate=rate, concurrency=concurrency)
    for pid, content in fetched.items():
        yield pid, content
