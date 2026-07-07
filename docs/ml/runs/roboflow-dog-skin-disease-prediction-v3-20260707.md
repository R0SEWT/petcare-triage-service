# Roboflow dog-skin-disease-prediction v3 acquisition — 2026-07-07

Task: `petcare-triage-service-ghn` · Status: acquired and deduped locally

## Source

- URL: https://universe.roboflow.com/majorproject-kopqr/dog-skin-disease-prediction
- Version/model: `dog-skin-disease-prediction/3`
- License shown by export: CC BY 4.0
- Export format: folder classification
- Export preprocessing: auto-orientation, resize to 640x640 stretch
- Raw local path: `ml/_raw/roboflow_dog_skin_disease_prediction_v3/` (gitignored)

## Raw Export Counts

Total: 361 images.

| Split | `flea_allergy` | `hotspot` | `mange` | `ringworm` |
| --- | ---: | ---: | ---: | ---: |
| train | 62 | 59 | 44 | 92 |
| val | 8 | 25 | 10 | 23 |
| test | 6 | 12 | 9 | 11 |

## PetCare Mapping

| Source class | PetCare mapping | Use |
| --- | --- | --- |
| `ringworm` | `dermatophytosis` | noisy supervised training/proxy |
| `flea_allergy` | `allergic_contact_dermatitis` | weak proxy only |
| `hotspot` | `unknown`, `unknown_ood` | dermatology OOD/proxy only |
| `mange` | `unknown`, `unknown_ood` | out-of-scope dermatology |

Prepared manifest:

- `ml/prepared/roboflow_dog_skin_disease_prediction_v3/manifest.jsonl`
- Rows: 361

## Dedup Result

Command used pHash threshold 4, kept train/val only, and referenced:

- `ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4/manifest.jsonl`
- `ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl`
- `ml/silver/silver-v0/manifest.jsonl`

Local dedup output:

- `ml/prepared/roboflow_dog_skin_disease_prediction_v3_dedup_phash4/manifest.jsonl`
- `ml/prepared/roboflow_dog_skin_disease_prediction_v3_dedup_phash4/removed.jsonl`
- `ml/prepared/roboflow_dog_skin_disease_prediction_v3_dedup_phash4/filter_report.json`

Rows:

| Status | Rows |
| --- | ---: |
| kept | 74 |
| removed | 287 |

Kept rows:

| Split | Class | Rows |
| --- | --- | ---: |
| train | `ringworm` | 9 |
| train | `hotspot` | 11 |
| train | `mange` | 30 |
| val | `hotspot` | 18 |
| val | `mange` | 6 |

Removed rows:

| Reason | Rows |
| --- | ---: |
| excluded test split | 38 |
| exact SHA duplicate | 4 |
| pHash near-duplicate | 245 |

The fallback is therefore much smaller after dedup than the raw export suggests:
only 9 independent-ish `ringworm` / `dermatophytosis` training candidates remain,
and no `flea_allergy` / `allergic_contact_dermatitis` proxy rows survived.

## Decision

Use the deduped output only as a tiny training/proxy supplement:

- add 9 noisy `dermatophytosis` candidates if we build a broader classifier;
- use 65 `unknown_ood` dermatology examples for OOD/proxy testing;
- do not report metrics from this source as clinical accuracy;
- do not publish public artifacts until provenance/license risk is revisited.
