import pytest

from scan_and_identify.config import Config, ConfigError


def test_config_from_env_reads_all_fields(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog.npz"
    catalog.write_bytes(b"")
    monkeypatch.setenv("SCAN_AND_IDENTIFY_API_KEY", "secret")
    monkeypatch.setenv("SCAN_AND_IDENTIFY_CATALOG_PATH", str(catalog))
    monkeypatch.setenv("R2_TCGPLAYER_BUCKET_ACCESS_KEY", "ak")
    monkeypatch.setenv("R2_TCGPLAYER_BUCKET_SECRET_KEY", "sk")
    monkeypatch.setenv("R2_TCGPLAYER_URL", "https://example.r2.cloudflarestorage.com")

    cfg = Config.from_env()

    assert cfg.api_key == "secret"
    assert cfg.catalog_path == catalog
    assert cfg.r2.access_key == "ak"
    assert cfg.r2.secret_key == "sk"
    assert cfg.r2.endpoint_url == "https://example.r2.cloudflarestorage.com"
    assert cfg.r2.bucket == "tcgplayerapi"
    assert cfg.tcgplayer_category == 1


def test_config_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SCAN_AND_IDENTIFY_API_KEY", raising=False)
    monkeypatch.setenv("SCAN_AND_IDENTIFY_CATALOG_PATH", "/nope")
    monkeypatch.setenv("R2_TCGPLAYER_BUCKET_ACCESS_KEY", "ak")
    monkeypatch.setenv("R2_TCGPLAYER_BUCKET_SECRET_KEY", "sk")
    monkeypatch.setenv("R2_TCGPLAYER_URL", "https://e.r2.cloudflarestorage.com")

    with pytest.raises(ConfigError, match="SCAN_AND_IDENTIFY_API_KEY"):
        Config.from_env()


def test_synthetic_fixtures_load(synthetic_parquets, synthetic_catalog):
    import numpy as np
    import pandas as pd

    products = pd.read_parquet(synthetic_parquets / "products.parquet")
    assert len(products) == 5
    assert "is_sealed" not in products.columns  # matches the real R2 schema

    data = np.load(synthetic_catalog, allow_pickle=False)
    assert data["embeddings"].shape == (4, 128)
    assert data["card_ids"].tolist() == ["1001", "1002", "1003", "2001"]
