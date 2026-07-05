import io

import httpx
import respx
from fastapi.testclient import TestClient
from httpx import Response
from PIL import Image

from scan_and_identify.server.app import create_app
from scan_and_identify.server.state import AppState


def _png_bytes() -> bytes:
    img = Image.new("RGB", (448, 448), (200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_identify_returns_candidates(synthetic_catalog, synthetic_parquets):
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    client = TestClient(create_app(state))
    url = "https://example.com/scan.png"
    with respx.mock(assert_all_called=True) as m:
        m.get(url).mock(return_value=Response(200, content=_png_bytes()))
        r = client.post(
            "/identify",
            json={"image_url": url, "top_k": 3, "rotation_invariant": False},
            headers={"Authorization": "Bearer k"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_card_back"] is False
    assert body["confidence"] in {"good", "fair", "poor"}
    assert len(body["candidates"]) == 3
    assert "product_id" in body["candidates"][0]


def test_identify_with_set_lock(synthetic_catalog, synthetic_parquets):
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    client = TestClient(create_app(state))
    url = "https://example.com/scan.png"
    with respx.mock(assert_all_called=True) as m:
        m.get(url).mock(return_value=Response(200, content=_png_bytes()))
        r = client.post(
            "/identify",
            json={"image_url": url, "set_ids": [100], "top_k": 5, "rotation_invariant": False},
            headers={"Authorization": "Bearer k"},
        )
    assert r.status_code == 200
    assert all(c["group_id"] == 100 for c in r.json()["candidates"])


def test_identify_with_set_ids_union(synthetic_catalog, synthetic_parquets):
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    client = TestClient(create_app(state))
    url = "https://example.com/scan.png"
    with respx.mock(assert_all_called=True) as m:
        m.get(url).mock(return_value=Response(200, content=_png_bytes()))
        r = client.post(
            "/identify",
            json={"image_url": url, "set_ids": [100, 200], "top_k": 5, "rotation_invariant": False},
            headers={"Authorization": "Bearer k"},
        )
    assert r.status_code == 200
    groups = {c["group_id"] for c in r.json()["candidates"]}
    assert groups <= {100, 200}


def test_identify_with_unknown_set_id_in_list_returns_404(synthetic_catalog, synthetic_parquets):
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    client = TestClient(create_app(state))
    url = "https://example.com/scan.png"
    with respx.mock(assert_all_called=True) as m:
        m.get(url).mock(return_value=Response(200, content=_png_bytes()))
        r = client.post(
            "/identify",
            json={"image_url": url, "set_ids": [100, 99999], "rotation_invariant": False},
            headers={"Authorization": "Bearer k"},
        )
    assert r.status_code == 404
    assert "99999" in r.json()["error"]["message"]


def test_identify_rejects_empty_set_ids(synthetic_catalog, synthetic_parquets):
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    client = TestClient(create_app(state))
    r = client.post(
        "/identify",
        json={"image_url": "https://example.com/x.png", "set_ids": []},
        headers={"Authorization": "Bearer k"},
    )
    assert r.status_code == 422  # Pydantic validation error


def test_identify_response_includes_printings_field(synthetic_catalog, synthetic_parquets):
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    client = TestClient(create_app(state))
    url = "https://example.com/scan.png"
    with respx.mock(assert_all_called=True) as m:
        m.get(url).mock(return_value=Response(200, content=_png_bytes()))
        r = client.post(
            "/identify",
            json={"image_url": url, "top_k": 1, "rotation_invariant": False},
            headers={"Authorization": "Bearer k"},
        )
    assert r.status_code == 200
    candidate = r.json()["candidates"][0]
    assert "printings" in candidate
    # Synthetic fixture: every product has both Normal and Foil SKUs.
    assert candidate["printings"] == ["Normal", "Foil"]


def test_identify_emits_telemetry_log_line(synthetic_catalog, synthetic_parquets, caplog):
    import logging

    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    client = TestClient(create_app(state))
    url = "https://example.com/scan.png"
    with respx.mock(assert_all_called=True) as m:
        m.get(url).mock(return_value=Response(200, content=_png_bytes()))
        with caplog.at_level(logging.INFO, logger="scan_and_identify.identify"):
            r = client.post(
                "/identify",
                json={"image_url": url, "top_k": 3, "rotation_invariant": False},
                headers={"Authorization": "Bearer k"},
            )
    assert r.status_code == 200
    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "scan_and_identify.identify"]
    assert len(msgs) == 1
    line = msgs[0]
    assert "identify" in line
    assert "top_pid=" in line
    assert "top_score=" in line
    assert "gap=" in line
    assert "conf=" in line
    assert "printings=" in line


def test_identify_image_fetch_404(synthetic_catalog, synthetic_parquets):
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    client = TestClient(create_app(state))
    url = "https://example.com/missing.png"
    with respx.mock() as m:
        m.get(url).mock(return_value=Response(404))
        r = client.post(
            "/identify",
            json={"image_url": url},
            headers={"Authorization": "Bearer k"},
        )
    assert r.status_code == 400
    assert "fetch" in r.json()["error"]["message"].lower()


def test_identify_uses_apps_shared_http_client(synthetic_catalog, synthetic_parquets):
    """The route must fetch through app.state.http_client — swap in a MockTransport
    client (no respx) and confirm the request never hits the real network."""
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    app = create_app(state)

    def handler(_request):
        return Response(200, content=_png_bytes())

    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TestClient(app)
    r = client.post(
        "/identify",
        json={"image_url": "https://imgs.test/scan.png", "top_k": 3, "rotation_invariant": False},
        headers={"Authorization": "Bearer k"},
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["candidates"]) == 3


def test_shared_http_client_closed_on_shutdown(synthetic_catalog, synthetic_parquets):
    state = AppState.bootstrap_for_tests("k", synthetic_catalog, synthetic_parquets)
    app = create_app(state)
    with TestClient(app):
        shared = app.state.http_client
        assert not shared.is_closed
    assert shared.is_closed
