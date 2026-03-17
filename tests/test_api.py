"""Tests for FastAPI endpoints."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import IndicatorResult, _results, app

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_service_name(self, client):
        data = client.get("/health").json()
        assert data["service"] == "miteco"
        assert data["status"] == "ok"


class TestCalculateEndpoint:
    def test_returns_202(self, client):
        resp = client.post("/calculate", json={"lau_code": "28079"})
        assert resp.status_code == 202

    def test_response_has_request_id(self, client):
        data = client.post("/calculate", json={"lau_code": "28079"}).json()
        assert "request_id" in data
        assert data["lau_code"] == "28079"

    def test_missing_lau_code_422(self, client):
        resp = client.post("/calculate", json={})
        assert resp.status_code == 422

    def test_with_custom_country_code(self, client):
        data = client.post(
            "/calculate", json={"lau_code": "1234", "country_code": "PT"}
        ).json()
        assert data["status"] == "accepted"


class TestIndicatorsEndpoint:
    def test_unknown_id_returns_404(self, client):
        resp = client.get("/indicators/nonexistent-id")
        assert resp.status_code == 404

    def test_returns_processing_for_in_progress(self, client):
        # Manually inject a result
        _results["test-123"] = IndicatorResult(
            request_id="test-123", lau_code="28079", status="processing"
        )
        try:
            resp = client.get("/indicators/test-123")
            assert resp.status_code == 200
            assert resp.json()["status"] == "processing"
        finally:
            _results.pop("test-123", None)

    def test_returns_completed_result(self, client):
        _results["test-456"] = IndicatorResult(
            request_id="test-456",
            lau_code="28079",
            municipality_name="Madrid",
            status="completed",
            zeu_area_m2=1_000_000,
            evu_m2=200_000,
            evu_ratio_pct=20.0,
            cau_m2=130_000,
            cau_ratio_pct=13.0,
        )
        try:
            resp = client.get("/indicators/test-456")
            data = resp.json()
            assert data["status"] == "completed"
            assert data["evu_m2"] == 200_000
            assert data["cau_m2"] == 130_000
        finally:
            _results.pop("test-456", None)
