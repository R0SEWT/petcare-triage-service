# Mendeley missing-label resolution

Date: 2026-07-07
Task: `petcare-triage-service-105`

## Decision

The missing labels in Mendeley dataset `5dbht54kw7` v1 cannot be recovered
deterministically from the public dataset metadata, paper page, supplementary
file, or local metadata currently available in the repository.

The 33 folders absent from `image_label.txt` must stay mapped as
`oodClass=unlabeled_source`, `condition=unknown`, and must not be used as
supervised training examples until a manual or veterinary adjudication pass
assigns labels to specific source folders.

## Confirmed Evidence

- Mendeley v1 reports 95 pet dogs and aggregate counts only: 23 bacterial
  dermatosis, 19 fungal infections, 23 hypersensitivity allergic dermatosis, and
  30 healthy. Source: https://data.mendeley.com/datasets/5dbht54kw7/1
- The Springer paper repeats those 95 dog-level counts and states that image
  labels were determined by veterinarians, with all data available in the
  Mendeley repository. Source: https://link.springer.com/article/10.1007/s13273-022-00249-7
- The Springer supplementary DOCX was checked on 2026-07-07. It contains
  figures and metric tables, but no `image_label` text, no `Dog210719...`
  folder IDs, and no complete per-folder label table.
- Local `folders.json` contains 95 source folders.
- Local `image_label.txt` contains labels for 62 source folders.
- Local `files.json` contains file inventory and folder IDs, but no disease
  labels.

## Audit Result

Command:

```bash
uv run python ml/audit_mendeley_labels.py
```

Observed folder counts in `image_label.txt`:

| Source label | Labeled folders |
| --- | ---: |
| `Bacterial_dermatosis` | 12 |
| `Fungal_infections` | 11 |
| `Hypersensitivity_allergic_dermatosis` | 13 |
| `Healthy` | 26 |

Aggregate missing counts inferred from paper totals:

| Source label | Missing folders |
| --- | ---: |
| `Bacterial_dermatosis` | 11 |
| `Fungal_infections` | 8 |
| `Hypersensitivity_allergic_dermatosis` | 10 |
| `Healthy` | 4 |

This aggregate distribution is useful for sanity-checking the dataset, but it
does not identify which of the 33 unlabeled source folders belongs to each
class.

## Adjudication Path

Generate a local review template:

```bash
uv run python ml/audit_mendeley_labels.py \
  --adjudication-csv ml/prepared/mendeley_5dbht54kw7_v1/missing-label-adjudication.csv
```

Reviewers should fill `adjudicated_source_label`, `adjudicated_condition`,
`adjudicated_ood_class`, `reviewer_id`, `reviewed_at`, and `notes`. Until then,
the unlabeled rows remain excluded from supervised training and all evaluation
claims.
