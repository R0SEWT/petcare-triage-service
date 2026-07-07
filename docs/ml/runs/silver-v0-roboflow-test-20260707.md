# silver-v0 Roboflow test proxy rehearsal — 2026-07-07

Task: `petcare-triage-service-c9n` · Status: local rehearsal complete

## Source

- Manifest: `ml/prepared/roboflow_dog_skin_disease_dataset_v2/manifest.jsonl`
- Split: `test`
- Source rows: 4,398
- Candidate test rows: 180
- Selected rows: 80
- Selection cap: 20 per available bucket
- Validation tier: proxy (`vetConfirmed=false`)

## Coverage

| Bucket | Rows |
| --- | ---: |
| `atopic_dermatitis` | 20 |
| `bacterial_pyoderma` | 20 |
| `fungal_malassezia` | 20 |
| `healthy_skin` | 20 |

Deficits at target 20:

| Bucket | Missing |
| --- | ---: |
| `dermatophytosis` | 20 |
| `allergic_contact_dermatitis` | 20 |
| `non_skin_pet` | 20 |
| `human_skin` | 20 |
| `other_species` | 20 |
| `poor_quality` | 20 |

## Leakage Checks

Compared against:

- `ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4/manifest.jsonl`
- `ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl`

Results:

| Check | Result |
| --- | --- |
| Exact SHA | clean: 0 matches |
| pHash threshold 4 | leakage found: 51 matches across 15 silver rows |

Interpretation: this is acceptable only as a process rehearsal. The held-out
Roboflow test split shares visual lineage with train/val augmentations, so
silver-v0 from this source is not an independent eval set.

## Local Artifacts

Generated under gitignored `ml/silver/`:

- `ml/silver/intake/silver-v0.roboflow-test.csv`
- `ml/silver/intake/silver-v0.roboflow-test.csv.report.json`
- `ml/silver/silver-v0/manifest.jsonl`
- `ml/silver/silver-v0/leakage-report-exact.json`
- `ml/silver/silver-v0/leakage-report-phash4.json`

Do not publish metrics from this rehearsal as clinical accuracy.
