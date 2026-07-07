# Triage Contract — executable artifacts (v0.1)

Machine-readable source of truth for the inference seam. The prose rationale is
in [`../inference-contract.md`](../inference-contract.md); these files are what
the mock service, the real service, and the frontend client all build against.

| File | Purpose | Consumed by |
| --- | --- | --- |
| `openapi.yaml` | Endpoint, multipart request, HTTP ↔ error-code mapping | mock + service (routing), frontend (client codegen) |
| `triage-response.schema.json` | 200 response body (JSON Schema 2020-12) | mock (output validation), frontend (response types), eval (record shape) |
| `triage-error.schema.json` | Non-200 error body | mock + service, frontend error handling |
| `labels.json` | Canonical condition taxonomy + localized display strings | service (class ids), frontend (localization), eval (label space) |
| `urgency-policy.json` | Ordered rule table: result → urgency class | service (urgency layer), documented for frontend |
| `gold-manifest.schema.json` | Frozen vet-verified eval manifest row | offline eval, HF dataset provenance, leakage checks |

## Design decisions baked into v0.1

- **`resultState` is explicit.** The response carries
  `completed | low_confidence | out_of_distribution | invalid` so the UI branches
  on one field instead of re-deriving state from confidence thresholds — that
  derivation is exactly the `Math.random()`/hardcoded-`87%` anti-pattern this
  contract removes.
- **Errors vs. outcomes are separated.** System/request failures are non-200
  (`triage-error.schema.json`). Low-confidence and out-of-distribution are
  *outcomes*, not errors: 200 responses with the matching `resultState`.
- **Canonical ids are authoritative.** Persisted records and model outputs use
  the `labels.json` `id`; display strings are presentation-only and match what
  PetCare-Web already renders (`src/lib/mock-data.ts`).
- **Urgency is policy, not model.** `urgency-policy.json` is an ordered rule
  table; it is never derived from raw confidence alone.

## Versioning

- The API contract is versioned in `openapi.yaml` (`info.version`, currently
  `0.1.0`). Breaking changes bump the minor while pre-1.0.
- `labels.json` and `urgency-policy.json` carry independent date-stamped
  `version` fields, surfaced in each response as `model.datasetVersion` /
  `model.calibrationVersion` lineage so persisted records stay auditable.
- Freeze v0.1 before creating `petcare-triage-service` (see project plan). Send
  this directory to the frontend owners for sign-off first.

## Validation (once tooling is added)

These are plain JSON Schema / OpenAPI 3.1 files with no custom extensions, so any
standard validator works (e.g. `ajv`, `check-jsonschema`, `redocly lint`). A
mock response validated against `triage-response.schema.json` is the acceptance
gate for Task 2 (mock service).
