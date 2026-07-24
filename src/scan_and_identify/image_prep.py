"""Shared image preprocessing for the embedder input.

Both the catalog build and the live query path MUST turn a card image into the
embedder's 448×448 input the *same* way. Milo was trained on 448×448 inputs and
ArcFace embeddings degrade when the aspect ratio is distorted, so if one path
letterboxes (aspect-preserving) while the other stretches, query and catalog
embeddings land in different distributions and match quality drops sharply.
Keeping this transform in one place is what prevents the two paths from drifting.
"""

from __future__ import annotations

from PIL import Image


def letterbox(image: Image.Image, size: int = 448) -> Image.Image:
    """Resize `image` to `size`×`size` RGB, preserving aspect ratio with white bars.

    The image is scaled so its longest side is `size`, then centered on a white
    `size`×`size` canvas. LANCZOS resampling keeps detail. This is the canonical
    embedder input transform — see the module docstring for why it must be shared.
    """
    img = image.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas
