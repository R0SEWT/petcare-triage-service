import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICES_DIR = Path(__file__).resolve().parents[1] / "services" / "triage-mock"
sys.path.insert(0, str(SERVICES_DIR))

from app import app  # noqa: E402
from capture import build_capture_row, capture_triage_best_effort  # noqa: E402


def response_payload() -> dict:
    return {
        "triageId": "triage_test",
        "resultState": "completed",
        "model": {"name": "mock", "version": "0", "datasetVersion": "labels", "calibrationVersion": "cal"},
        "imageQuality": {"isAcceptable": True, "issues": []},
        "ood": {"isOutOfDistribution": False, "score": 0.1, "reason": None},
        "prediction": {
            "condition": "atopic_dermatitis",
            "confidence": 0.8,
            "urgency": "follow_up",
            "topK": [{"condition": "atopic_dermatitis", "confidence": 0.8}],
        },
    }


def test_capture_row_hashes_ids_and_omits_free_text():
    row = build_capture_row(
        capture_id="cap_test",
        captured_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
        image_rel_path=Path("images/cap_test.jpg"),
        image_bytes=b"image",
        content_type="image/jpeg",
        pet_id="raw-pet-id",
        species="dog",
        body_region="belly",
        pet_age_months=24,
        breed="mixed",
        symptom_notes="owner phone maybe here",
        client_request_id="raw-request-id",
        response=response_payload(),
    )

    encoded = json.dumps(row, sort_keys=True)
    assert "raw-pet-id" not in encoded
    assert "raw-request-id" not in encoded
    assert "owner phone maybe here" not in encoded
    assert row["petIdHash"]
    assert row["clientRequestIdHash"]
    assert row["request"]["symptomNotesPresent"] is True
    assert row["feedback"]["vetConfirmedCondition"] is None


def test_capture_best_effort_skips_without_consent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PETCARE_CAPTURE_BUFFER_DIR", str(tmp_path))

    result = capture_triage_best_effort(
        consent_for_model_improvement=False,
        image_bytes=b"image",
        content_type="image/jpeg",
        pet_id="pet",
        species="dog",
        body_region="belly",
        pet_age_months=None,
        breed=None,
        symptom_notes=None,
        client_request_id=None,
        response=response_payload(),
    )

    assert result.captured is False
    assert result.reason == "no_consent"
    assert not (tmp_path / "metadata.jsonl").exists()


def test_capture_best_effort_writes_image_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PETCARE_CAPTURE_BUFFER_DIR", str(tmp_path))

    result = capture_triage_best_effort(
        consent_for_model_improvement=True,
        image_bytes=b"image",
        content_type="image/jpeg",
        pet_id="pet",
        species="dog",
        body_region="belly",
        pet_age_months=None,
        breed=None,
        symptom_notes=None,
        client_request_id=None,
        response=response_payload(),
    )

    assert result.captured is True
    assert result.image_path is not None
    assert result.image_path.exists()
    rows = [json.loads(line) for line in (tmp_path / "metadata.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["image"]["sha256"]
    assert rows[0]["sync"]["hfDatasetRepo"] is None


def test_analyze_capture_failure_does_not_block_response(monkeypatch: pytest.MonkeyPatch):
    def fail_capture(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.capture_triage_best_effort", fail_capture)
    client = TestClient(app)
    response = client.post(
        "/api/triage/analyze",
        files={"image": ("skin.jpg", b"image", "image/jpeg")},
        data={
            "petId": "pet",
            "species": "dog",
            "bodyRegion": "belly",
            "consentForModelImprovement": "true",
            "mockScenario": "completed",
        },
    )

    assert response.status_code == 200
