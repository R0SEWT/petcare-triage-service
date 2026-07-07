"""In-process smoke test: every scenario returns the right status and a
schema-valid body. Run: python smoke_test.py"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["PETCARE_CAPTURE_BUFFER_DIR"] = tempfile.mkdtemp(prefix="petcare-capture-smoke-")

from app import app
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "ml" / "schema"
RESP = Draft202012Validator(json.loads((SCHEMA_DIR / "triage-response.schema.json").read_text()))
ERR = Draft202012Validator(json.loads((SCHEMA_DIR / "triage-error.schema.json").read_text()))

client = TestClient(app)


def post(scenario, content_type="image/jpeg", size=1024, species="dog"):
    files = {"image": ("s.jpg", io.BytesIO(b"x" * size), content_type)}
    data = {"petId": "1", "species": species, "bodyRegion": "belly",
            "consentForModelImprovement": "true", "mockScenario": scenario}
    return client.post("/api/triage/analyze", files=files, data=data)


CASES = [
    ("completed", 200, "completed"),
    ("completed_urgent", 200, "completed"),
    ("low_confidence", 200, "low_confidence"),
    ("out_of_distribution", 200, "out_of_distribution"),
    ("poor_quality", 422, None),
    ("unsupported_species", 422, None),
    ("invalid_type", 400, None),
    ("too_large", 413, None),
    ("timeout", 504, None),
    ("unavailable", 503, None),
]

failures = []
for scenario, want_status, want_state in CASES:
    r = post(scenario)
    body = r.json()
    if r.status_code != want_status:
        failures.append(f"{scenario}: status {r.status_code} != {want_status}")
        continue
    if want_status == 200:
        errs = list(RESP.iter_errors(body))
        if errs:
            failures.append(f"{scenario}: 200 body invalid: {errs[0].message}")
        elif body["resultState"] != want_state:
            failures.append(f"{scenario}: resultState {body['resultState']} != {want_state}")
    else:
        errs = list(ERR.iter_errors(body))
        if errs:
            failures.append(f"{scenario}: error body invalid: {errs[0].message}")

# Real-validation paths independent of scenario
r = post("completed", content_type="image/gif")
if r.status_code != 400:
    failures.append(f"gif upload: expected 400, got {r.status_code}")
r = post("completed", species="hamster")
if r.status_code != 422:
    failures.append(f"hamster species: expected 422, got {r.status_code}")
r = post("completed", size=6 * 1024 * 1024)
if r.status_code != 413:
    failures.append(f"6MB upload: expected 413, got {r.status_code}")

if failures:
    print("FAIL")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"OK — {len(CASES)} scenarios + 3 validation paths all pass")
