# Roboflow dog-skin-disease-prediction v3

Task: `petcare-triage-service-ghn` · Status: fallback ready, download pending API key

Source: https://universe.roboflow.com/majorproject-kopqr/dog-skin-disease-prediction

## Why This Source

Kaivlya `dogs-skin-disease-fxh4x` has better taxonomy coverage, but its public
Roboflow page reports `Dataset versions 0` and `Models 0`, so there is no stable
Universe export version. This dataset is smaller but versioned:

- License shown by Roboflow: CC BY 4.0
- Task: classification
- Images: 361
- Dataset versions: 3
- Current model/version: `dog-skin-disease-prediction/3`
- Classes: `flea_allergy`, `hotspot`, `mange`, `ringworm`

## Mapping

| Source class | PetCare mapping | Use |
| --- | --- | --- |
| `ringworm` | `dermatophytosis` | noisy supervised training/proxy |
| `flea_allergy` | `allergic_contact_dermatitis` | weak proxy only |
| `hotspot` | `unknown`, `unknown_ood` | dermatology OOD/proxy only |
| `mange` | `unknown`, `unknown_ood` | out-of-scope dermatology |

These labels are not vet-confirmed and must never enter `gold-v0`.

## Download

The current environment does not have `ROBOFLOW_API_KEY`, `roboflow` CLI, or the
Roboflow Python package configured. When the key is available in the shell, use
the official CLI or SDK to download the folder/classification export into
gitignored raw storage.

CLI shape from Roboflow docs:

```bash
uvx roboflow download \
  -f folder \
  -l ml/_raw/roboflow_dog_skin_disease_prediction_v3 \
  majorproject-kopqr/dog-skin-disease-prediction/3
```

Do not commit the raw export.

## Prepare

```bash
uv run python ml/prepare_roboflow_prediction.py \
  --source-dir ml/_raw/roboflow_dog_skin_disease_prediction_v3 \
  --out ml/prepared/roboflow_dog_skin_disease_prediction_v3
```

Then dedup before use:

```bash
uv run --with pillow --with imagehash python ml/filter_roboflow_dedup.py \
  --candidate-manifest ml/prepared/roboflow_dog_skin_disease_prediction_v3/manifest.jsonl \
  --reference-manifest ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4/manifest.jsonl \
  --reference-manifest ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl \
  --out ml/prepared/roboflow_dog_skin_disease_prediction_v3_dedup_phash4
```

If `ml/silver/silver-v0/manifest.jsonl` or future `ml/gold/gold-v0/manifest.jsonl`
exists locally, include them as additional reference manifests.
