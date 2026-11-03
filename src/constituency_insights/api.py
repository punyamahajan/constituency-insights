"""FastAPI inference service with startup-only model loading and text-free logs."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any

import joblib
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constituency_insights.nlp.distilbert import LABEL_ORDER


class StanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2000)
    target: str = Field(min_length=1, max_length=200)

    @field_validator("text", "target")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(min_length=1, max_length=32)

    @field_validator("features")
    @classmethod
    def finite_features(cls, value: dict[str, float]) -> dict[str, float]:
        import math
        if any(not math.isfinite(number) for number in value.values()):
            raise ValueError("features must contain finite numbers")
        return value


class StanceResponse(BaseModel):
    label: str
    probabilities: dict[str, float]


class OutcomeResponse(BaseModel):
    predicted_share: float
    conformal_interval: tuple[float, float]


class ModelService:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: dict[str, Any] = {}
        self.stance_model: Any | None = None
        self.stance_tokenizer: Any | None = None
        self.outcome_model: Any | None = None
        self.conformal_quantile: float | None = None
        self.load_count = 0
        self.downloaded_checkpoint: Path | None = None

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load(self) -> None:
        if self.load_count:
            return
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        stance = self.config["stance"]
        checkpoint = self._resolve_checkpoint(Path(stance["checkpoint"]), stance)
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Stance checkpoint missing: {weights}")
        if self._hash(weights) != stance["checkpoint_sha256"]:
            raise ValueError("Stance checkpoint hash does not match serving config")
        self.stance_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.stance_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True
        ).to("cpu").eval()
        outcome = self.config.get("outcome", {})
        artifact = Path(outcome.get("artifact", ""))
        if not artifact.is_file() and os.getenv("OUTCOME_GCS_OBJECT"):
            artifact = self._download_gcs_file(
                os.environ["MODEL_GCS_BUCKET"], os.environ["OUTCOME_GCS_OBJECT"],
                os.getenv("OUTCOME_MODEL_SHA256"), ".joblib",
            )
        if artifact.is_file():
            self.outcome_model = joblib.load(artifact)
            metadata = Path(outcome.get("conformal", ""))
            if not metadata.is_file() and os.getenv("OUTCOME_CONFORMAL_OBJECT"):
                metadata = self._download_gcs_file(
                    os.environ["MODEL_GCS_BUCKET"], os.environ["OUTCOME_CONFORMAL_OBJECT"],
                    os.getenv("OUTCOME_CONFORMAL_SHA256"), ".json",
                )
            if not metadata.is_file():
                raise FileNotFoundError(f"Conformal metadata missing: {metadata}")
            self.conformal_quantile = float(json.loads(metadata.read_text(encoding="utf-8"))["quantile"])
        self.load_count += 1

    def _download_gcs_file(self, bucket: str, object_name: str,
                           expected_hash: str | None, suffix: str) -> Path:
        from google.cloud import storage
        descriptor, filename = tempfile.mkstemp(prefix="constituency-artifact-", suffix=suffix)
        os.close(descriptor)
        destination = Path(filename)
        storage.Client().bucket(bucket).blob(object_name).download_to_filename(destination)
        if expected_hash and self._hash(destination) != expected_hash.lower():
            raise ValueError(f"Downloaded GCS artifact hash mismatch: {object_name}")
        return destination

    def _resolve_checkpoint(self, configured: Path, stance: dict[str, Any]) -> Path:
        """Download a GCS archive at startup when configured, then return its directory."""
        bucket = os.getenv("MODEL_GCS_BUCKET")
        object_name = os.getenv("MODEL_GCS_OBJECT")
        if not bucket and not object_name:
            return configured
        if not bucket or not object_name:
            raise RuntimeError("MODEL_GCS_BUCKET and MODEL_GCS_OBJECT must be set together")
        from google.cloud import storage
        root = Path(tempfile.mkdtemp(prefix="constituency-model-"))
        archive = root / "checkpoint.tar.gz"
        storage.Client().bucket(bucket).blob(object_name).download_to_filename(archive)
        expected_archive_hash = os.getenv("MODEL_GCS_ARCHIVE_SHA256")
        if expected_archive_hash and self._hash(archive) != expected_archive_hash.lower():
            raise ValueError("Downloaded GCS model archive hash does not match MODEL_GCS_ARCHIVE_SHA256")
        extract_root = root / "checkpoint"
        extract_root.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            root_resolved = extract_root.resolve()
            for member in tar.getmembers():
                target = (extract_root / member.name).resolve()
                if (root_resolved not in target.parents or member.issym() or member.islnk()
                        or not (member.isfile() or member.isdir())):
                    raise ValueError("Unsafe path in downloaded model archive")
            tar.extractall(extract_root)
        candidates = list(extract_root.rglob("model.safetensors"))
        if len(candidates) != 1:
            raise FileNotFoundError("GCS model archive must contain exactly one model.safetensors")
        checkpoint = candidates[0].parent
        if self._hash(candidates[0]) != str(stance["checkpoint_sha256"]).lower():
            raise ValueError("Downloaded checkpoint hash does not match serving config")
        self.downloaded_checkpoint = checkpoint
        return checkpoint

    def stance(self, request: StanceRequest) -> StanceResponse:
        if self.stance_model is None or self.stance_tokenizer is None:
            raise HTTPException(status_code=503, detail="Stance model unavailable")
        encoded = self.stance_tokenizer(
            [request.text], text_pair=[request.target], padding=True,
            truncation=True, max_length=int(self.config["stance"]["max_length"]),
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.stance_model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0].tolist()
        values = {label: float(probabilities[index]) for index, label in enumerate(LABEL_ORDER)}
        return StanceResponse(label=max(values, key=values.get), probabilities=values)

    def outcome(self, request: OutcomeRequest) -> OutcomeResponse:
        if self.outcome_model is None or self.conformal_quantile is None:
            raise HTTPException(status_code=503, detail="Outcome model artifact is unavailable")
        order = self.config["outcome"]["feature_order"]
        missing = sorted(set(order) - set(request.features))
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing outcome features: {missing}")
        prediction = float(self.outcome_model.predict([[request.features[name] for name in order]])[0])
        radius = self.conformal_quantile
        return OutcomeResponse(predicted_share=prediction,
                               conformal_interval=(max(0.0, prediction - radius), min(100.0, prediction + radius)))

    def info(self) -> dict[str, Any]:
        stance = self.config.get("stance", {})
        checkpoint = self.downloaded_checkpoint or Path(stance.get("checkpoint", ""))
        weights = checkpoint / "model.safetensors"
        outcome_artifact = Path(self.config.get("outcome", {}).get("artifact", ""))
        import transformers
        return {"service_version": "0.1.0", "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "stance_checkpoint_sha256": self._hash(weights) if weights.is_file() else None,
                "outcome_artifact_sha256": self._hash(outcome_artifact) if outcome_artifact.is_file() else None,
                "stance_training_date": stance.get("training_date"),
                "outcome_training_date": self.config.get("outcome", {}).get("training_date"),
                "models_loaded": self.load_count > 0}


def create_app(config_path: Path | str = Path("configs/serving.yaml"), service: ModelService | None = None) -> FastAPI:
    model_service = service or ModelService(Path(config_path))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_service.load()
        app.state.model_service = model_service
        yield

    app = FastAPI(title="Constituency Insights Inference API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model-info")
    async def model_info() -> dict[str, Any]:
        return model_service.info()

    @app.post("/predict-stance", response_model=StanceResponse)
    async def predict_stance(request: StanceRequest) -> StanceResponse:
        return model_service.stance(request)

    @app.post("/predict-outcome", response_model=OutcomeResponse)
    async def predict_outcome(request: OutcomeRequest) -> OutcomeResponse:
        return model_service.outcome(request)

    return app


app = create_app()
