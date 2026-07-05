"""FastAPI app factory."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from scan_and_identify.image_fetch import FetchError, fetch_image
from scan_and_identify.pipeline import IdentifyResult
from scan_and_identify.server.auth import require_bearer
from scan_and_identify.server.schemas import (
    ExportRequest,
    IdentifyBatchRequest,
    IdentifyBatchResponse,
    IdentifyRequest,
    IdentifyResponse,
    ResolveSkuRequest,
    SearchResponse,
)
from scan_and_identify.server.state import AppState
from scan_and_identify.tcgplayer.seller_csv import MergePriceConflict, build_seller_csv

log = logging.getLogger("scan_and_identify.identify")
export_log = logging.getLogger("scan_and_identify.export")


def _candidate_dicts(candidates) -> list[dict]:
    return [c.__dict__ for c in candidates]


def _log_identify(
    *,
    image_url: str,
    scan_id: str | None,
    set_ids: list[int] | None,
    result: IdentifyResult,
    fetch_ms: float,
    pipeline_ms: float,
) -> None:
    """One grep-friendly line per identify call.

    Format (space-separated key=value), so ``docker logs | awk '{...}'``
    can tally tiers / pids without parsing JSON::

        identify scan_id=quill-42 url=https://... card_back=False
                 top_pid=218276 top_score=0.717 gap=0.347 conf=good
                 n_candidates=3 set_ids=[2655] printings=[Normal,Foil]
                 fetch_ms=87 pipeline_ms=412

    Empty / None fields show as ``-`` so the column count stays stable.
    """
    if result.candidates:
        top = result.candidates[0]
        top_pid = top.product_id
        top_score = f"{top.score:.4f}"
        gap = (
            top.score - result.candidates[1].score if len(result.candidates) > 1 else top.score
        )
        gap_str = f"{gap:.4f}"
        printings = "[" + ",".join(top.printings) + "]"
    else:
        top_pid = "-"
        top_score = "-"
        gap_str = "-"
        printings = "-"
    set_ids_str = "[" + ",".join(str(s) for s in set_ids) + "]" if set_ids else "-"
    log.info(
        "identify scan_id=%s url=%s card_back=%s top_pid=%s top_score=%s gap=%s conf=%s "
        "n_candidates=%d set_ids=%s printings=%s fetch_ms=%d pipeline_ms=%d",
        scan_id or "-",
        image_url,
        result.is_card_back,
        top_pid,
        top_score,
        gap_str,
        result.confidence or "-",
        len(result.candidates),
        set_ids_str,
        printings,
        round(fetch_ms),
        round(pipeline_ms),
    )


def create_app(state: AppState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.http_client.aclose()

    app = FastAPI(title="scan-and-identify", version="0.1.0", lifespan=lifespan)
    auth = require_bearer(state.api_key)
    app.state.deps = state
    # One pooled client for all image fetches — created here (not in lifespan
    # startup) so tests that skip the lifespan still get it; closed on shutdown.
    app.state.http_client = httpx.AsyncClient(timeout=10.0)

    @app.exception_handler(HTTPException)
    async def http_exc(_request, exc: HTTPException):
        code = "http_" + str(exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": exc.detail}},
        )

    router = APIRouter(dependencies=[Depends(auth)])

    @router.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "catalog_version": state.catalog_version,
            "catalog_built_at": state.catalog_built_at,
            "catalog_size": len(state.catalog),
            "parquet_synced_at": state.parquet_synced_at.isoformat(),
        }

    @router.get("/sets")
    async def sets() -> dict:
        return {"sets": state.store.set_list()}

    @router.get("/search", response_model=SearchResponse)
    async def search(
        name: str | None = None,
        collector_number: str | None = None,
        set_ids: list[int] | None = Query(default=None),
        limit: int = 20,
    ) -> dict:
        if not name and not collector_number:
            raise HTTPException(
                status_code=400,
                detail="At least one of 'name' or 'collector_number' is required",
            )
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
        try:
            results = state.store.search_products(
                name=name,
                collector_number=collector_number,
                set_ids=set_ids,
                limit=limit,
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        # store.product() returns "tcgplayer_url" + "clean_name" + "is_sealed" — strip to ProductMatch shape.
        out = []
        for p in results:
            out.append(
                {
                    "product_id": p["product_id"],
                    "name": p["name"],
                    "set_name": p["set_name"] or "",
                    "set_abbr": p["set_abbr"] or "",
                    "group_id": p["group_id"],
                    "collector_number": p["collector_number"],
                    "rarity": p["rarity"],
                    "image_url": p["image_url"] or "",
                    "printings": state.store.printings_for_product(p["product_id"]),
                }
            )
        return {"results": out}

    @router.post("/identify", response_model=IdentifyResponse)
    async def identify(req: IdentifyRequest) -> dict:
        t0 = time.perf_counter()
        try:
            image = await fetch_image(req.image_url, client=app.state.http_client)
        except FetchError as e:
            raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}") from e
        t1 = time.perf_counter()
        try:
            # pipeline.identify is sync (Milo ONNX + numpy + pHash). Offload to a
            # thread so the asyncio event loop stays responsive to other requests.
            # The default thread pool is bounded (~12 workers on mtg-eye); concurrent
            # batches will queue rather than oversubscribe the CPU.
            result = await asyncio.to_thread(
                state.pipeline.identify,
                image=image,
                set_ids=req.set_ids,
                top_k=req.top_k,
                rotation_invariant=req.rotation_invariant,
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        t2 = time.perf_counter()
        _log_identify(
            image_url=req.image_url,
            scan_id=None,
            set_ids=req.set_ids,
            result=result,
            fetch_ms=(t1 - t0) * 1000,
            pipeline_ms=(t2 - t1) * 1000,
        )
        return {
            "is_card_back": result.is_card_back,
            "confidence": result.confidence,
            "candidates": _candidate_dicts(result.candidates),
        }

    @router.post("/identify-batch", response_model=IdentifyBatchResponse)
    async def identify_batch(req: IdentifyBatchRequest) -> dict:
        async def one(item):
            t0 = time.perf_counter()
            try:
                image = await fetch_image(item.image_url, client=app.state.http_client)
            except FetchError as e:
                return {
                    "id": item.id,
                    "is_card_back": False,
                    "confidence": None,
                    "candidates": [],
                    "error": f"fetch failed: {e}",
                }
            t1 = time.perf_counter()
            try:
                # See /identify above for the rationale. With N=58 items in a batch
                # and a 12-worker pool, expect ~5× wall-clock speedup vs serial.
                result = await asyncio.to_thread(
                    state.pipeline.identify,
                    image=image,
                    set_ids=req.set_ids,
                    top_k=req.top_k,
                    rotation_invariant=req.rotation_invariant,
                )
            except KeyError as e:
                return {
                    "id": item.id,
                    "is_card_back": False,
                    "confidence": None,
                    "candidates": [],
                    "error": str(e),
                }
            t2 = time.perf_counter()
            _log_identify(
                image_url=item.image_url,
                scan_id=item.id,
                set_ids=req.set_ids,
                result=result,
                fetch_ms=(t1 - t0) * 1000,
                pipeline_ms=(t2 - t1) * 1000,
            )
            return {
                "id": item.id,
                "is_card_back": result.is_card_back,
                "confidence": result.confidence,
                "candidates": _candidate_dicts(result.candidates),
                "error": None,
            }

        results = await asyncio.gather(*(one(i) for i in req.images))
        return {"results": list(results)}

    @router.get("/products/{product_id}")
    async def get_product(product_id: int) -> dict:
        p = state.store.product(product_id)
        if p is None:
            raise HTTPException(status_code=404, detail=f"Unknown product_id {product_id}")
        p["skus"] = state.store.skus_for_product(product_id)
        return p

    @router.post("/products/{product_id}/resolve-sku")
    async def resolve_sku(product_id: int, req: ResolveSkuRequest) -> dict:
        if state.store.product(product_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown product_id {product_id}")
        sku = state.store.resolve_sku(
            product_id,
            printing=req.printing,
            condition=req.condition,
            language=req.language,
        )
        if sku is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No SKU for product {product_id} with printing={req.printing}, "
                    f"condition={req.condition}, language={req.language}"
                ),
            )
        return {
            "sku_id": sku["sku_id"],
            "market_price": sku["market_price"],
            "low_price": sku["low_price"],
            "mid_price": sku["mid_price"],
            "high_price": sku["high_price"],
            "direct_low_price": sku["direct_low_price"],
        }

    @router.post("/export/tcgplayer-csv")
    async def export_csv(req: ExportRequest) -> Response:
        t0 = time.perf_counter()
        rows = [r.model_dump() for r in req.rows]
        formula = req.price_formula.model_dump() if req.price_formula else None
        try:
            body = build_seller_csv(
                state.store,
                rows,
                merge_duplicates=req.merge_duplicates,
                price_formula=formula,
            )
        except MergePriceConflict as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "merge_price_conflict",
                        "message": str(e),
                        "conflicts": e.conflicts,
                    }
                },
            )
        export_log.info(
            "export rows_in=%d rows_out=%d merge_duplicates=%s duration_ms=%d",
            len(rows),
            body.count(b"\n") - 1,  # minus header line
            req.merge_duplicates,
            round((time.perf_counter() - t0) * 1000),
        )
        return Response(
            content=body,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="tcgplayer-export.csv"'},
        )

    app.include_router(router)
    return app
