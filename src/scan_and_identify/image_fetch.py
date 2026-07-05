"""Fetch images from URLs into PIL Images."""

from __future__ import annotations

import io

import httpx
from PIL import Image, UnidentifiedImageError


class FetchError(RuntimeError):
    pass


async def fetch_image(
    url: str, timeout: float = 10.0, client: httpx.AsyncClient | None = None
) -> Image.Image:
    """Fetch ``url`` and decode it as an RGB PIL Image.

    Pass a long-lived ``client`` to reuse pooled connections (the server holds
    one on ``app.state.http_client``); it is never closed here. Without one, an
    ephemeral client is created per call — fine for scripts and tests.
    """
    if client is None:
        async with httpx.AsyncClient(timeout=timeout) as ephemeral:
            resp = await ephemeral.get(url)
    else:
        resp = await client.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise FetchError(f"GET {url} returned {resp.status_code}")
    try:
        img = Image.open(io.BytesIO(resp.content))
        img.load()
    except UnidentifiedImageError as e:
        raise FetchError(f"Response from {url} is not a valid image") from e
    return img.convert("RGB")
