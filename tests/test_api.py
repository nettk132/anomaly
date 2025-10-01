from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint(temp_env):
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_project_crud_minimal(temp_env):
    with TestClient(app) as client:
        resp = client.get("/projects")
        assert resp.status_code == 200
        assert resp.json() == []

        payload = {"name": "demo", "description": "", "training_mode": "anomaly"}
        create = client.post("/projects", json=payload)
        assert create.status_code == 200
        created = create.json()
        assert created["name"] == "demo"
        pid = created["project_id"]

        got = client.get(f"/projects/{pid}")
        assert got.status_code == 200
        assert got.json()["project_id"] == pid

        listing = client.get("/projects")
        assert listing.status_code == 200
        assert len(listing.json()) == 1

        # Delete requires confirmation string (project name)
        delete = client.delete(f"/projects/{pid}", params={"confirm": "demo"})
        assert delete.status_code == 200
