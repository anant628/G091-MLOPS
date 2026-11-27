import pytest
from fastapi.testclient import TestClient

from app import main

VALID = {
    "patients_ahead": 50,
    "doctors_on_duty": 6,
    "arrival_hour": 11,
    "avg_consult_min": 10.5,
    "lab_tests_pending": 15,
    "distance_km": 20.0,
    "department": "General",
    "wing": "Old Wing",
}


class DummyModel:
    def predict(self, X):
        return [42.0]


@pytest.fixture
def client(monkeypatch):
    # swap in a fake model so tests never need models/model.joblib
    monkeypatch.setattr(main, "get_model", lambda: DummyModel())
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.json()["prediction"] == 42.0


def test_distance_above_70_rejected(client):
    r = client.post("/predict", json={**VALID, "distance_km": 71})
    assert r.status_code == 422


def test_negative_lab_tests_rejected(client):
    r = client.post("/predict", json={**VALID, "lab_tests_pending": -1})
    assert r.status_code == 422


def test_boundary_values_accepted(client):
    r = client.post("/predict", json={**VALID, "distance_km": 70, "lab_tests_pending": 0})
    assert r.status_code == 200


def test_extra_fields_rejected_before_model(client, monkeypatch):
    def unexpected_model_load():
        pytest.fail("Invalid requests must not reach the model")

    monkeypatch.setattr(main, "get_model", unexpected_model_load)
    r = client.post("/predict", json={**VALID, "record_id": "R1"})
    assert r.status_code == 422


def test_missing_field_rejected_before_model(client, monkeypatch):
    def unexpected_model_load():
        pytest.fail("Invalid requests must not reach the model")

    monkeypatch.setattr(main, "get_model", unexpected_model_load)
    body = {key: value for key, value in VALID.items() if key != "wing"}
    assert client.post("/predict", json=body).status_code == 422
