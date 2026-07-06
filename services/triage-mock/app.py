"""Contract-conformant mock for the PetCare triage inference service.

Implements POST /api/triage/analyze exactly as specified in
docs/ml/schema/ (contract v0.1). It serves canned-but-schema-valid responses so
PetCare-Web can integrate against a real seam before any model exists — replacing
the client-side Math.random()/hardcoded-87% flow.

Design notes:
- No model. Responses are deterministic, driven by an explicit scenario selector
  (form field `mockScenario` or header `X-Mock-Scenario`), NOT randomness. That
  is the whole point: the frontend chooses which state to render.
- Urgency is computed from docs/ml/schema/urgency-policy.json, never hardcoded.
- Every 200 response is validated against triage-response.schema.json before it
  leaves the server, so the mock can never drift from the contract.
- This folder is self-contained and meant to be lifted into
  `petcare-triage-service` once the contract is frozen and a baseline exists.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Form, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "ml" / "schema"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}
ALLOWED_SPECIES = {"dog", "cat"}

# --- load the contract artifacts (source of truth) ---------------------------
LABELS = json.loads((SCHEMA_DIR / "labels.json").read_text())
URGENCY_POLICY = json.loads((SCHEMA_DIR / "urgency-policy.json").read_text())
RESPONSE_VALIDATOR = Draft202012Validator(
    json.loads((SCHEMA_DIR / "triage-response.schema.json").read_text())
)

SEVERITY_BY_ID = {l["id"]: l.get("clinicalSeverity", "unknown") for l in LABELS["labels"]}
DISPLAY_ES = {l["id"]: l["display"]["es"] for l in LABELS["labels"]}

MODEL = {
    "name": "petcare-derm-yolov8-cls",  # YOLOv8 classification mode — see docs/ml/model-task-decision.md
    "version": "0.0.0-mock",
    "datasetVersion": LABELS["version"],
    "calibrationVersion": URGENCY_POLICY["version"],
}

app = FastAPI(title="PetCare Triage Mock", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; the real service scopes this to the app origin
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- urgency policy evaluator (reads urgency-policy.json) ---------------------
def confidence_band(conf: float) -> str:
    bands = URGENCY_POLICY["confidenceBands"]
    if conf >= bands["high"]["min"]:
        return "high"
    if conf >= bands["medium"]["min"]:
        return "medium"
    return "low"


def resolve_urgency(result_state: str, condition: Optional[str], conf: Optional[float]) -> str:
    severity = SEVERITY_BY_ID.get(condition or "", "unknown")
    band = confidence_band(conf) if conf is not None else None
    for rule in URGENCY_POLICY["rules"]:
        when = rule["when"]
        if "resultState" in when and when["resultState"] != result_state:
            continue
        if "clinicalSeverity" in when and when["clinicalSeverity"] != severity:
            continue
        if "confidenceBand" in when and band not in when["confidenceBand"]:
            continue
        return rule["urgency"]
    return "follow_up"  # matches urgency-policy.json default rule


# --- response builders -------------------------------------------------------
def _error(status: int, code: str, message: str, retryable: bool, req_id: Optional[str]):
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "retryable": retryable, "clientRequestId": req_id}},
    )


def _completed_body(result_state: str, condition: str, conf: float, topk: list[tuple[str, float]],
                    latency: int, req_id: Optional[str]) -> dict[str, Any]:
    urgency = resolve_urgency(result_state, condition, conf)
    return {
        "triageId": f"triage_mock{int(time.time() * 1000)}",
        "status": "completed",
        "resultState": result_state,
        "clientRequestId": req_id,
        "model": MODEL,
        "latencyMs": latency,
        "imageQuality": {"isAcceptable": True, "issues": []},
        "ood": {"isOutOfDistribution": False, "score": 0.08, "reason": None},
        "prediction": {
            "condition": condition,
            "displayName": DISPLAY_ES[condition],
            "confidence": conf,
            "urgency": urgency,
            "topK": [{"condition": c, "confidence": p} for c, p in topk],
        },
        "recommendations": [
            "Evita el contacto con posibles irritantes.",
            "Consulta con un veterinario si empeora o persiste.",
        ],
        "safety": {
            "message": "PetCare ofrece triaje preventivo, no diagnostico medico.",
            "requiresVeterinarian": urgency in ("consult_soon", "urgent"),
        },
    }


def _ood_body(req_id: Optional[str]) -> dict[str, Any]:
    return {
        "triageId": f"triage_mock{int(time.time() * 1000)}",
        "status": "completed",
        "resultState": "out_of_distribution",
        "clientRequestId": req_id,
        "model": MODEL,
        "latencyMs": 900,
        "imageQuality": {"isAcceptable": True, "issues": []},
        "ood": {"isOutOfDistribution": True, "score": 0.94, "reason": "not_pet_skin"},
        "prediction": None,
        "recommendations": ["Vuelve a tomar la foto siguiendo la guia de fotografia."],
        "safety": {
            "message": "No pudimos analizar esta imagen. PetCare no reemplaza al veterinario.",
            "requiresVeterinarian": False,
        },
    }


# Scenario -> builder. Deterministic. Default is a confident completed result.
def build_200(scenario: str, req_id: Optional[str]) -> Optional[dict[str, Any]]:
    if scenario == "completed":
        return _completed_body(
            "completed", "allergic_contact_dermatitis", 0.87,
            [("allergic_contact_dermatitis", 0.87), ("bacterial_pyoderma", 0.06), ("atopic_dermatitis", 0.04)],
            1840, req_id,
        )
    if scenario == "completed_urgent":
        return _completed_body(
            "completed", "bacterial_pyoderma", 0.83,
            [("bacterial_pyoderma", 0.83), ("fungal_malassezia", 0.09), ("dermatophytosis", 0.05)],
            2100, req_id,
        )
    if scenario == "low_confidence":
        return _completed_body(
            "low_confidence", "atopic_dermatitis", 0.34,
            [("atopic_dermatitis", 0.34), ("allergic_contact_dermatitis", 0.28), ("fungal_malassezia", 0.19)],
            1600, req_id,
        )
    if scenario == "out_of_distribution":
        return _ood_body(req_id)
    return None


ERROR_SCENARIOS = {
    "invalid_type": (400, "invalid_image_type", "Formato invalido. Solo JPG o PNG.", False),
    "too_large": (413, "image_too_large", "El archivo supera el limite de 5MB.", False),
    "poor_quality": (422, "poor_image_quality", "Imagen borrosa u oscura. Vuelve a tomarla.", False),
    "unsupported_species": (422, "unsupported_species", "Especie fuera de alcance.", False),
    "timeout": (504, "model_timeout", "El analisis excedio el tiempo limite.", True),
    "unavailable": (503, "model_unavailable", "Servicio no disponible. Reintenta mas tarde.", True),
}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL["name"], "contract": "v0.1"}


@app.post("/api/triage/analyze")
async def analyze(
    image: UploadFile = File(...),
    petId: str = Form(...),
    species: str = Form(...),
    bodyRegion: str = Form(...),
    consentForModelImprovement: bool = Form(...),
    petAgeMonths: Optional[int] = Form(None),
    breed: Optional[str] = Form(None),
    symptomNotes: Optional[str] = Form(None),
    clientRequestId: Optional[str] = Form(None),
    mockScenario: Optional[str] = Form(None),
    x_mock_scenario: Optional[str] = Header(None),
):
    scenario = mockScenario or x_mock_scenario or "completed"

    # Forced error scenarios (for exercising UI failure states deterministically).
    if scenario in ERROR_SCENARIOS:
        status, code, msg, retry = ERROR_SCENARIOS[scenario]
        return _error(status, code, msg, retry, clientRequestId)

    # Real input validation the server always enforces (client checks are advisory).
    if image.content_type not in ALLOWED_TYPES:
        return _error(400, "invalid_image_type", f"MIME no soportado: {image.content_type}", False, clientRequestId)
    body = await image.read()
    if len(body) > MAX_IMAGE_BYTES:
        return _error(413, "image_too_large", "El archivo supera el limite de 5MB.", False, clientRequestId)
    if species not in ALLOWED_SPECIES:
        return _error(422, "unsupported_species", f"Especie fuera de alcance: {species}", False, clientRequestId)

    payload = build_200(scenario, clientRequestId)
    if payload is None:
        return _error(422, "poor_image_quality", f"Escenario desconocido: {scenario}", False, clientRequestId)

    # Self-check: never emit a body that violates the contract.
    errors = sorted(RESPONSE_VALIDATOR.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        return JSONResponse(status_code=500, content={"error": {
            "code": "model_unavailable",
            "message": "Mock produced a non-conformant response: " + errors[0].message,
            "retryable": True, "clientRequestId": clientRequestId,
        }})
    return JSONResponse(status_code=200, content=payload)
