from fastapi.testclient import TestClient

from scan_and_identify.server.app import create_app
from scan_and_identify.server.state import AppState


def test_sets_returns_sorted_by_name(synthetic_catalog, synthetic_parquets):
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    client = TestClient(create_app(state))
    r = client.get("/sets", headers={"Authorization": "Bearer k"})
    assert r.status_code == 200
    sets = r.json()["sets"]
    assert [s["abbr"] for s in sets] == ["TSA", "TSB"]
    assert "is_current" not in sets[0]
