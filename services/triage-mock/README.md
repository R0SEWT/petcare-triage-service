# triage-mock

Standalone, contract-conformant mock of the PetCare triage inference service.
Serves `POST /api/triage/analyze` per [`../../docs/ml/schema/`](../../docs/ml/schema/)
(contract v0.1) so PetCare-Web can integrate against a real seam before any model
exists — replacing the client-side `Math.random()` / hardcoded-`87%` flow.

This directory is self-contained and designed to be lifted into
`petcare-triage-service` once the contract is frozen and a baseline model exists.

## Run

```bash
cd services/triage-mock
uv venv && source .venv/bin/activate      # or: python3 -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Health check: `GET http://localhost:8000/health`

## Contract

- Every 200 body is validated against `triage-response.schema.json` before it is
  returned — the mock cannot drift from the contract.
- Urgency is computed from `urgency-policy.json`, never hardcoded.
- Model name is `petcare-derm-yolov8-cls` (see `../../docs/ml/model-task-decision.md`).
- If `consentForModelImprovement=true`, successful analyses are written
  best-effort to the local capture buffer documented in
  `../../docs/ml/data/consented-capture-buffer.md`.

## Deterministic scenarios (no randomness)

Choose the response with form field `mockScenario` or header `X-Mock-Scenario`.
Default is `completed`. This is how the frontend drives each UI state without
`Math.random()`:

| Scenario | HTTP | `resultState` / error code |
| --- | ---: | --- |
| `completed` (default) | 200 | `completed` — confident allergic contact dermatitis |
| `completed_urgent` | 200 | `completed` — bacterial pyoderma → `consult_soon` |
| `low_confidence` | 200 | `low_confidence` |
| `out_of_distribution` | 200 | `out_of_distribution` (prediction is null) |
| `poor_quality` | 422 | `poor_image_quality` |
| `unsupported_species` | 422 | `unsupported_species` |
| `invalid_type` | 400 | `invalid_image_type` |
| `too_large` | 413 | `image_too_large` |
| `timeout` | 504 | `model_timeout` |
| `unavailable` | 503 | `model_unavailable` |

Real validation always runs regardless of scenario: non-JPEG/PNG → 400,
> 5 MB → 413, species not dog/cat → 422.

## Example

```bash
curl -s -X POST http://localhost:8000/api/triage/analyze \
  -F image=@sample.jpg -F petId=1 -F species=dog -F bodyRegion=belly \
  -F consentForModelImprovement=true -F mockScenario=completed | jq .
```

## Test

`python smoke_test.py` starts the app in-process (Starlette TestClient) and
asserts every scenario returns the right status and a schema-valid body.
