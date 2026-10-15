from scan_and_identify.tcgplayer.store import TCGStore


def test_store_loads_sets_sorted_by_name(synthetic_parquets):
    store = TCGStore.load(synthetic_parquets)
    sets = store.set_list()
    assert [s["abbr"] for s in sets] == ["TSA", "TSB"]
    assert sets[0] == {"group_id": 100, "name": "Test Set Alpha", "abbr": "TSA"}


def test_store_product_returns_full_row(synthetic_parquets):
    store = TCGStore.load(synthetic_parquets)
    p = store.product(1001)
    assert p["product_id"] == 1001
    assert p["name"] == "Alpha Card 1"
    assert p["set_name"] == "Test Set Alpha"
    assert p["set_abbr"] == "TSA"
    assert p["rarity"] == "Common"
    assert p["collector_number"] == "1"
    assert p["is_sealed"] is False


def test_store_derives_sealed_from_unopened_sku(synthetic_parquets):
    store = TCGStore.load(synthetic_parquets)
    assert store.product(9999)["is_sealed"] is True
    found = [p["product_id"] for p in store.search_products(name="Alpha")]
    assert found == [1001, 1002, 1003]  # "Alpha Sealed Box" (9999) excluded


def test_store_skus_for_product_joins_lookups(synthetic_parquets):
    store = TCGStore.load(synthetic_parquets)
    skus = store.skus_for_product(1001)
    assert len(skus) == 4
    s0 = skus[0]
    assert s0["printing"] in {"Normal", "Foil"}
    assert s0["condition"] in {"Near Mint", "Lightly Played"}
    assert s0["language"] == "English"
    assert isinstance(s0["market_price"], (float, type(None)))


def test_store_resolve_sku_returns_matching_sku(synthetic_parquets):
    store = TCGStore.load(synthetic_parquets)
    sku = store.resolve_sku(1001, printing="Normal", condition="Near Mint", language="English")
    assert sku is not None
    assert sku["product_id"] == 1001
    assert sku["printing"] == "Normal"
    assert sku["condition"] == "Near Mint"
    assert sku["language"] == "English"


def test_store_resolve_sku_returns_none_when_missing(synthetic_parquets):
    store = TCGStore.load(synthetic_parquets)
    assert (
        store.resolve_sku(1001, printing="Normal", condition="Damaged", language="English") is None
    )


def test_printings_for_product_returns_normal_then_foil(synthetic_parquets):
    store = TCGStore.load(synthetic_parquets)
    # All synthetic products have both printing_id=1 (Normal) and 2 (Foil) SKUs.
    assert store.printings_for_product(1001) == ["Normal", "Foil"]


def test_printings_for_product_empty_for_unknown(synthetic_parquets):
    store = TCGStore.load(synthetic_parquets)
    assert store.printings_for_product(7777777) == []
