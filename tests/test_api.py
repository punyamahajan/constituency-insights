from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import logging

from constituency_insights.api import ModelService, OutcomeResponse, StanceResponse, create_app


class FakeService(ModelService):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.config = {"stance": {"training_date": "2026-01-01"}, "outcome": {"training_date": "2026-01-02"}}

    def load(self) -> None:
        self.load_count += 1

    def stance(self, request):
        return StanceResponse(label="FAVOR", probabilities={"AGAINST": .1, "FAVOR": .8, "NONE": .1})

    def outcome(self, request):
        return OutcomeResponse(predicted_share=52.0, conformal_interval=(45.0, 59.0))


def test_health_model_info_and_startup_load_once(tmp_path: Path) -> None:
    service = FakeService(tmp_path / "unused.yaml")
    with TestClient(create_app(service=service)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/model-info").json()["models_loaded"] is True
        assert client.get("/model-info").json()["stance_training_date"] == "2026-01-01"
    assert service.load_count == 1


def test_prediction_endpoints_and_invalid_inputs(tmp_path: Path) -> None:
    with TestClient(create_app(service=FakeService(tmp_path / "unused.yaml"))) as client:
        assert client.post("/predict-stance", json={"text": "support", "target": "Policy"}).json()["label"] == "FAVOR"
        assert client.post("/predict-outcome", json={"features": {"x": 1.0}}).json()["conformal_interval"] == [45.0, 59.0]
        assert client.post("/predict-stance", json={"text": " ", "target": "x"}).status_code == 422
        assert client.post("/predict-stance", json={"text": "x"}).status_code == 422
        assert client.post("/predict-stance", json={"text": "x", "target": "x", "extra": 1}).status_code == 422
        assert client.post("/predict-outcome", json={"features": {"x": "not-number"}}).status_code == 422


def test_outcome_unavailable_is_explicit_503(tmp_path: Path) -> None:
    service = FakeService(tmp_path / "unused.yaml")
    with TestClient(create_app(service=service)) as client:
        service.outcome_model = None
        service.outcome = ModelService.outcome.__get__(service, ModelService)
        assert client.post("/predict-outcome", json={"features": {"x": 1}}).status_code == 503


def test_request_text_never_appears_in_application_logs(tmp_path: Path, caplog) -> None:
    secret = "UNIQUE-REQUEST-TEXT-MUST-NOT-LEAK"
    service = FakeService(tmp_path / "unused.yaml")
    with caplog.at_level(logging.DEBUG):
        with TestClient(create_app(service=service)) as client:
            assert client.post("/predict-stance", json={"text": secret, "target": "Policy"}).status_code == 200
    assert secret not in "\n".join(record.getMessage() for record in caplog.records)
