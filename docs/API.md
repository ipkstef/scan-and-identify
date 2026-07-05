# API

Stateless HTTP API. All endpoints require:

```
Authorization: Bearer $SCAN_AND_IDENTIFY_API_KEY
```

Requests and responses are JSON, except `POST /export/tcgplayer-csv` which
returns `text/csv`. Errors have the shape:

```json
{"error": {"code": "string", "message": "string"}}
```

The `code` is `http_<status>` for generic errors; specific types like
`merge_price_conflict` use a domain code and may include extra fields.

---

## `GET /health`

```json
{
  "status": "ok",
  "catalog_version": "2026-05",
  "catalog_built_at": "2026-05-19T00:00:00Z",
  "catalog_size": 111100,
  "parquet_synced_at": "2026-05-17T07:47:41.990282+00:00"
}
```

`catalog_version` is `YYYY-MM` derived from `catalog_built_at`. `parquet_synced_at`
is when the in-memory `TCGStore` was last loaded (= container boot time; no
hot-reload in v1).

---

## `GET /sets`

All TCGplayer sets for the configured category, current sets first.

```json
{
  "sets": [
    {"group_id": 24234, "name": "Commander: Tarkir: Dragonstorm", "abbr": "TDC", "is_current": true},
    {"group_id": 23445, "name": "Commander: Modern Horizons 3", "abbr": "M3C", "is_current": true}
  ]
}
```

Pass `group_id`s as `set_ids` on `/identify` to lock identification to one or
more sets.

---

## `POST /identify`

```json
{
  "image_url": "https://your-storage/scans/scan42.jpg",
  "set_ids": [24234],
  "top_k": 5,
  "rotation_invariant": true
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `image_url` | string | required | HTTP(S), reachable from the container, returns an image. |
| `set_ids` | list[int] \| null | null | Locks candidates to the union of these sets. `[24234]` for one set, `[24234, 23445]` for multiple. 404 if any id is unknown; 422 if empty. |
| `top_k` | int | 5 | 1–20. |
| `rotation_invariant` | bool | true | Embeds the 180° rotation as well and keeps the better top match — but only when the 0° top score is below the early-exit threshold (0.5), so right-side-up scans pay no extra latency. |

Response:

```json
{
  "is_card_back": false,
  "confidence": "good",
  "candidates": [
    {
      "product_id": 591234,
      "score": 0.93,
      "name": "Lightning Bolt",
      "set_name": "Tarkir: Dragonstorm",
      "set_abbr": "TDC",
      "group_id": 24234,
      "collector_number": "146",
      "rarity": "Common",
      "image_url": "https://tcgplayer-cdn.tcgplayer.com/product/591234_200w.jpg",
      "printings": ["Normal", "Foil"]
    }
  ]
}
```

`printings` is a subset of `["Normal", "Foil"]` — the SKUs this `product_id`
is sold in. Foil variants like "Galaxy Foil" / "Etched Foil" are separate
`product_id`s, not separate printings; check the product `name` suffix for
those.

Errors:

- `400 Could not fetch image: ...` — image URL unreachable or non-image.
- `404 Unknown set_id N` — some id in `set_ids` doesn't exist.
- `422` — `set_ids` was an empty list (use `null` for "no lock").

---

## `POST /identify-batch`

```json
{
  "images": [
    {"id": "scan42", "image_url": "..."},
    {"id": "scan43", "image_url": "..."}
  ],
  "set_ids": [24234],
  "top_k": 5,
  "rotation_invariant": true
}
```

`id` is opaque — echoed back. Per-image errors don't fail the batch; the
response always has one entry per input, in order.

```json
{
  "results": [
    {"id": "scan42", "is_card_back": false, "confidence": "good", "candidates": [...], "error": null},
    {"id": "scan43", "is_card_back": false, "confidence": "poor", "candidates": [...], "error": null},
    {"id": "scan44", "is_card_back": false, "confidence": null, "candidates": [], "error": "fetch failed: ..."}
  ]
}
```

---

## `GET /search`

Product search for the correction UI.

| Param | Type | Notes |
|---|---|---|
| `name` | string | Case-insensitive substring of the product name. |
| `collector_number` | string | Case-insensitive exact match. |
| `set_ids` | list[int] | Optional union filter. Repeat the param: `?set_ids=24234&set_ids=23445`. 404 on unknown. |
| `limit` | int | 1–100, default 20. |

At least one of `name` or `collector_number` is required. Sealed products
are excluded.

Same response shape as `/identify` candidates, minus `score`.

---

## `GET /products/{product_id}`

Full product + all SKU variants.

```json
{
  "product_id": 591234,
  "name": "Lightning Bolt",
  "clean_name": "Lightning Bolt",
  "group_id": 24234,
  "set_name": "Tarkir: Dragonstorm",
  "set_abbr": "TDC",
  "collector_number": "146",
  "rarity": "Common",
  "is_sealed": false,
  "image_url": "https://tcgplayer-cdn.tcgplayer.com/product/591234_200w.jpg",
  "tcgplayer_url": "https://www.tcgplayer.com/product/591234/...",
  "skus": [
    {
      "sku_id": 9049325,
      "product_id": 591234,
      "printing": "Normal",
      "condition": "Near Mint",
      "language": "English",
      "market_price": 0.45,
      "low_price": 0.10,
      "mid_price": 0.30,
      "high_price": 1.99,
      "direct_low_price": 0.15
    }
  ]
}
```

`404` if `product_id` is unknown.

---

## `POST /products/{product_id}/resolve-sku`

```json
{"printing": "Normal", "condition": "Near Mint", "language": "English"}
```

Allowed values:

- `printing`: `Normal`, `Foil`
- `condition`: `Near Mint`, `Lightly Played`, `Moderately Played`, `Heavily Played`, `Damaged`
- `language`: `English`, `Chinese (S)`, `Chinese (T)`, `French`, `German`, `Italian`, `Japanese`, `Korean`, `Portuguese`, `Russian`, `Spanish`

```json
{
  "sku_id": 9049325,
  "market_price": 0.45,
  "low_price": 0.10,
  "mid_price": 0.30,
  "high_price": 1.99,
  "direct_low_price": 0.15
}
```

`404` if the product or the variant combination doesn't exist. Prices may
be `null` when TCGplayer has no marketplace data.

---

## `POST /export/tcgplayer-csv`

Render the TCGplayer seller-bulk-upload CSV.

```json
{
  "rows": [
    {
      "product_id": 591234,
      "printing": "Normal",
      "condition": "Near Mint",
      "language": "English",
      "quantity": 1,
      "marketplace_price": 0.49
    }
  ],
  "merge_duplicates": true,
  "price_formula": {
    "reference": "market",
    "modifier": {"type": "percent", "value": 2.0}
  }
}
```

| Field | Notes |
|---|---|
| `rows[].marketplace_price` | Per-row override. Wins over `price_formula`. |
| `merge_duplicates` | Default `true`. Sums quantities for rows sharing `(product_id, printing, condition, language)`. |
| `price_formula.reference` | One of `market`, `low`, `mid`, `high`, `direct_low`. |
| `price_formula.modifier.type` | `percent` or `fixed`. |
| `price_formula.modifier.value` | E.g., `2.0` for +2%, `-0.01` for -1¢. |

Returns `200 text/csv` with `Content-Disposition: attachment;
filename="tcgplayer-export.csv"`. Columns match TCGplayer's seller template:

```
TCGplayer Id, Product Line, Set Name, Product Name, Title, Number, Rarity,
Condition, TCG Market Price, TCG Direct Low, TCG Low Price With Shipping,
TCG Low Price, Total Quantity, Add to Quantity, TCG Marketplace Price, Photo URL
```

If `merge_duplicates=true` and two rows for the same SKU specify different
`marketplace_price`, the request is rejected:

```json
{
  "error": {
    "code": "merge_price_conflict",
    "message": "merge_duplicates found 1 price conflict(s)",
    "conflicts": [
      {
        "sku_key": {"product_id": 1001, "printing": "Normal", "condition": "Near Mint", "language": "English"},
        "prices_seen": [1.49, 1.99]
      }
    ]
  }
}
```

Reconcile prices and resubmit, or pass `merge_duplicates: false`. Rows that
don't resolve to a real SKU are emitted with blank `TCGplayer Id` and price
columns; TCGplayer's importer rejects them individually, the rest of the
file succeeds.

`Product Line` is hardcoded to `Magic` in v1.

---

## Confidence

The `score` field is `0.85 · cosine + 0.15 · (1 − hamming/64)` where
`cosine` is the Milo embedding similarity and `hamming` is the bit distance
between the input's name-region pHash and the catalog's reference pHash for
that product. Always in `[0, 1]`.

`gap` is `top_1.score - top_2.score`. Tiers default to:

| Tier | Rule |
|---|---|
| `good` | `score ≥ 0.55 AND gap ≥ 0.15` |
| `poor` | `score < 0.45 OR gap < 0.05` |
| `fair` | everything else |

Override via env vars: `SCAN_AND_IDENTIFY_CONF_{GOOD,POOR}_{SCORE,GAP}`.
