# Roboflow Universe smoke test

Task: `petcare-triage-service-1zk`
Status: live API smoke test completed.

## Goal

Run pretrained Roboflow Universe dog dermatology models against a small,
fixed PetCare probe set before downloading any Roboflow datasets. This keeps
storage use low and quickly tells us whether the models are directionally useful
for bootstrapping.

## Harness

Script:

```bash
uv run python ml/roboflow_smoke_test.py --dry-run
```

Live run:

```bash
export ROBOFLOW_API_KEY=...
uv run python ml/roboflow_smoke_test.py
```

The script:

- reads `ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl`
- samples Mendeley probes from:
  - `bacterial_pyoderma`
  - `fungal_malassezia`
  - `atopic_dermatitis`
  - `healthy_skin`
  - `unlabeled_source`
- preserves expected label, source label, OOD class, and SHA-256 in each result
- sends probes only when `ROBOFLOW_API_KEY` exists
- writes outputs under `ml/runs/roboflow_smoke/<timestamp>/`, which is ignored
  by git
- never prints the API key

Extra OOD probes can be added without changing the script:

```bash
uv run python ml/roboflow_smoke_test.py \
  --extra-image non_skin:/path/to/non_skin_pet.jpg \
  --extra-image human_skin:/path/to/human_skin.jpg
```

## Default models

| Name | Task | Model ID | Purpose |
| --- | --- | --- | --- |
| taxonomy-aligned classifier | classification | `dog-skin-disease-dataset/2` | primary go/no-go model |
| clinical-named classifier | classification | `dog-skin-disease-prediction/3` | second-opinion taxonomy probe |
| lesion detector | object detection | `dog-skin-diseases/1` | optional localization/crop signal |

## Live result: 2026-07-06

Run output:

- local output directory: `ml/runs/roboflow_smoke/20260706-195503/`
- probes: 10 Mendeley images
- requests: 30 total, 10 per model
- HTTP result: 30/30 succeeded with status 200
- latency: 642 ms min, 1,030 ms average, 4,075 ms max

The first live attempt returned HTTP 400 because Roboflow expected base64 image
payloads. The harness was corrected to send base64-encoded image bodies.

Top predictions:

| Probe bucket | `dog-skin-disease-dataset/2` | `dog-skin-disease-prediction/3` | `dog-skin-diseases/1` |
| --- | --- | --- | --- |
| `bacterial_pyoderma` | `healthy`, `healthy` | `ringworm`, `mange` | `bacterial-dermatosis`, `bacterial-dermatosis` |
| `fungal_malassezia` | `healthy`, `healthy` | `ringworm`, `ringworm` | `bacterial-dermatosis`, `bacterial-dermatosis` |
| `atopic_dermatitis` | `healthy`, `hypersensitivity dermatitis` | `mange`, `ringworm` | `bacterial-dermatosis`, no detection |
| `healthy_skin` | `healthy`, `healthy` | `mange`, `ringworm` | `healthy`, `bacterial-dermatosis` |
| `unlabeled_source` | `healthy`, `healthy` | `ringworm`, `hotspot` | `healthy`, no detection |

Decision: the hosted pretrained Roboflow models are useful for integration
testing, but they are not reliable enough to reuse as PetCare model behavior.
The taxonomy-aligned classifier over-predicts `healthy` on Mendeley disease
probes. The clinical-named classifier over-predicts `ringworm`/`mange`, including
healthy probes. The detector finds broad lesion boxes but over-calls
`bacterial-dermatosis`.

This does not fully rule out Roboflow datasets as noisy training data. It does
mean we should train our own model from exported images and manifests rather
than relying on hosted Universe model predictions or page metrics.

## Go/no-go criteria

Original go/no-go criteria:

- `dog-skin-disease-dataset/2` gives plausible labels on labeled Mendeley probes
- healthy and unlabeled Mendeley probes do not collapse into a single disease
  class with high confidence
- obvious non-skin pet images are rejected or low confidence once extra probes
  are added
- latency/cost of API probing is acceptable for development use

Do not use Roboflow metrics as PetCare performance. Any downloaded/exported
dataset must still go through license capture, manifesting, and perceptual
deduplication before supervised training.

Updated recommendation: do not export every candidate yet. If we proceed with
Roboflow mirrors, export the primary taxonomy-aligned dataset first, preserve all
metadata and `Unlabeled` images, and treat it as noisy bootstrap training data
for a PetCare-owned baseline.
