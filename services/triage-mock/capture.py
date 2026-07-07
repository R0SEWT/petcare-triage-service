"""Best-effort consented capture buffer for triage requests."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUFFER_DIR = REPO_ROOT / "data" / "bronze" / "triage-captures"
SCHEMA_VERSION = "petcare-capture-buffer-v0"
CONSENT_SCOPE = "model_improvement"
SOURCE_SERVICE = "triage-mock"

CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


@dataclass(frozen=True)
class CaptureResult:
    captured: bool
    reason: str
    capture_id: str | None = None
    metadata_path: Path | None = None
    image_path: Path | None = None


def capture_enabled() -> bool:
    raw = os.getenv("PETCARE_CAPTURE_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def buffer_dir() -> Path:
    configured = os.getenv("PETCARE_CAPTURE_BUFFER_DIR")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_BUFFER_DIR


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_optional(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def image_extension(content_type: str) -> str:
    return CONTENT_TYPE_EXT.get(content_type, ".bin")


def safe_prediction_summary(response: dict[str, Any]) -> dict[str, Any] | None:
    prediction = response.get("prediction")
    if prediction is None:
        return None
    return {
        "condition": prediction.get("condition"),
        "confidence": prediction.get("confidence"),
        "urgency": prediction.get("urgency"),
        "topK": prediction.get("topK", []),
    }


def dataset_card() -> str:
    return """---
license: other
task_categories:
- image-classification
pretty_name: PetCare Consented Triage Captures
---

# PetCare consented triage captures

Private dataset buffer synced from the PetCare triage service after explicit
model-improvement consent. Rows are raw app captures, not vet-confirmed labels,
and must not be used for evaluation or clinical accuracy claims without a
separate review/adjudication process.
"""


def ensure_dataset_card(root: Path) -> None:
    card = root / "README.md"
    if not card.exists():
        card.write_text(dataset_card(), encoding="utf-8")


def build_capture_row(
    *,
    capture_id: str,
    captured_at: datetime,
    image_rel_path: Path,
    image_bytes: bytes,
    content_type: str,
    pet_id: str,
    species: str,
    body_region: str,
    pet_age_months: int | None,
    breed: str | None,
    symptom_notes: str | None,
    client_request_id: str | None,
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "captureId": capture_id,
        "capturedAt": captured_at.isoformat(),
        "sourceService": SOURCE_SERVICE,
        "triageId": response["triageId"],
        "clientRequestIdHash": hash_optional(client_request_id),
        "petIdHash": hash_optional(pet_id),
        "consent": {
            "modelImprovement": True,
            "scope": CONSENT_SCOPE,
        },
        "image": {
            "path": image_rel_path.as_posix(),
            "contentType": content_type,
            "sizeBytes": len(image_bytes),
            "sha256": sha256_bytes(image_bytes),
        },
        "request": {
            "species": species,
            "bodyRegion": body_region,
            "petAgeMonths": pet_age_months,
            "breed": breed,
            "symptomNotesPresent": bool(symptom_notes),
            "symptomNotesLength": len(symptom_notes or ""),
        },
        "triage": {
            "resultState": response["resultState"],
            "model": response["model"],
            "imageQuality": response["imageQuality"],
            "ood": response["ood"],
            "prediction": safe_prediction_summary(response),
        },
        "feedback": {
            "helpful": None,
            "ownerCorrection": None,
            "vetConfirmedCondition": None,
            "outcomeNotes": None,
        },
        "sync": {
            "hfDatasetRepo": None,
            "syncedAt": None,
        },
    }


def write_capture(
    *,
    root: Path,
    image_bytes: bytes,
    content_type: str,
    pet_id: str,
    species: str,
    body_region: str,
    pet_age_months: int | None,
    breed: str | None,
    symptom_notes: str | None,
    client_request_id: str | None,
    response: dict[str, Any],
) -> CaptureResult:
    captured_at = utc_now()
    capture_id = f"cap_{time.time_ns()}_{uuid4().hex[:8]}"
    date_path = captured_at.strftime("%Y/%m/%d")
    ext = image_extension(content_type)
    image_rel_path = Path("images") / date_path / f"{capture_id}{ext}"
    image_path = root / image_rel_path
    metadata_path = root / "metadata.jsonl"

    root.mkdir(parents=True, exist_ok=True)
    ensure_dataset_card(root)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)

    row = build_capture_row(
        capture_id=capture_id,
        captured_at=captured_at,
        image_rel_path=image_rel_path,
        image_bytes=image_bytes,
        content_type=content_type,
        pet_id=pet_id,
        species=species,
        body_region=body_region,
        pet_age_months=pet_age_months,
        breed=breed,
        symptom_notes=symptom_notes,
        client_request_id=client_request_id,
        response=response,
    )
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return CaptureResult(True, "captured", capture_id, metadata_path, image_path)


def capture_triage_best_effort(
    *,
    consent_for_model_improvement: bool,
    image_bytes: bytes,
    content_type: str,
    pet_id: str,
    species: str,
    body_region: str,
    pet_age_months: int | None,
    breed: str | None,
    symptom_notes: str | None,
    client_request_id: str | None,
    response: dict[str, Any],
) -> CaptureResult:
    if not consent_for_model_improvement:
        return CaptureResult(False, "no_consent")
    if not capture_enabled():
        return CaptureResult(False, "disabled")
    try:
        return write_capture(
            root=buffer_dir(),
            image_bytes=image_bytes,
            content_type=content_type,
            pet_id=pet_id,
            species=species,
            body_region=body_region,
            pet_age_months=pet_age_months,
            breed=breed,
            symptom_notes=symptom_notes,
            client_request_id=client_request_id,
            response=response,
        )
    except Exception:
        LOGGER.exception("failed to write consented triage capture")
        return CaptureResult(False, "write_failed")
