# Roboflow Universe smoke test

Task: `petcare-triage-service-1zk`
Status: harness ready; live API run pending `ROBOFLOW_API_KEY`.

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

## Current local result

As of 2026-07-06, `ROBOFLOW_API_KEY` was not set in the shell environment, so
the live API run was not executed. The dry-run path validates local probe
selection and can be run without secrets.

Expected next command after loading a key:

```bash
uv run python ml/roboflow_smoke_test.py
```

## Go/no-go criteria

Proceed to controlled dataset export only if:

- `dog-skin-disease-dataset/2` gives plausible labels on labeled Mendeley probes
- healthy and unlabeled Mendeley probes do not collapse into a single disease
  class with high confidence
- obvious non-skin pet images are rejected or low confidence once extra probes
  are added
- latency/cost of API probing is acceptable for development use

Do not use Roboflow metrics as PetCare performance. Any downloaded/exported
dataset must still go through license capture, manifesting, and perceptual
deduplication before supervised training.
