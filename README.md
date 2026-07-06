# petcare-triage-service

ML / backend for the PetCare dermatological triage demo. Owns the inference
**contract**, a conformant **mock**, and the **model** work. The frontend
(`PetCare-Web`) consumes the contract over HTTP via an env-configured URL and
contains no clinical logic.

Split out of `PetCare-Web` once the contract froze (v0.1) and the baseline began
— the trigger named in the original architecture plan — to keep ML work
decoupled from a frontend repo it does not own.

## Layout

| Path | What |
| --- | --- |
| `docs/ml/` | The seam. Prose contract + executable `schema/` (OpenAPI + JSON Schema + `labels.json` + `urgency-policy.json`), model-task decision, and data plan. Source of truth for both sides. |
| `services/triage-mock/` | Standalone FastAPI mock of `POST /api/triage/analyze`. Deterministic scenarios, urgency from the policy file, every 200 body schema-validated. `python smoke_test.py`. |
| `ml/` | YOLOv8-cls baseline (v0, coarse). `prepare_data.py` (stdlib) + `train_yolo_cls.py` (GPU box). See `ml/README.md`. |

## Status

- Contract v0.1 — frozen, pending frontend-owner sign-off.
- Mock — verified end-to-end over HTTP.
- Baseline — training project ready; runs on a Lightning AI GPU box.
- Credible accuracy still requires the vet-verified gold set
  (`docs/ml/data/gold-eval-set.md`); v0 numbers are directional.
