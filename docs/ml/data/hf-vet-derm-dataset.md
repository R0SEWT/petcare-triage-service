# HF `shrayyyy/vet-derm-dataset` evaluation

Task: `petcare-triage-service-x51`
Date: 2026-07-07

## Source

- HF dataset: https://huggingface.co/datasets/shrayyyy/vet-derm-dataset
- HF revision inspected: `69ed4ac2f4ee5abbc4eca18d7e5d9f01835d63d9`
- Declared license: Apache-2.0
- Declared lineage: Kaggle
  `youssefmohmmed/dogs-skin-diseases-image-dataset`
- HF files: `README.md`, `paligemma_dataset.jsonl`, `images.zip`

Local raw files are gitignored under `ml/_raw/hf_vet_derm_dataset/`.

Downloaded checksums:

| File | SHA-256 |
| --- | --- |
| `README.md` | `e1482ce6399009a550ed85f0f6b6f7275dfcf01f4b70a4f40f7285b9a442d5f2` |
| `paligemma_dataset.jsonl` | `b14341138fdd3787ccd44ac2533cc7f282822c2cf5936bb5643ef889c01ad3e9` |
| `images.zip` | `e22be9cd3d38c5d1910268e75d5cc1ca8e17612cbee7652d0344a4d22ab371bb` |

## Raw Counts

`paligemma_dataset.jsonl` contains 3,882 rows:

| Source class | Train | Val | Total |
| --- | ---: | ---: | ---: |
| `Dermatitis` | 546 | 175 | 721 |
| `Fungal_infections` | 375 | 97 | 472 |
| `Healthy` | 492 | 139 | 631 |
| `Hypersensitivity` | 230 | 63 | 293 |
| `demodicosis` | 588 | 174 | 762 |
| `ringworm` | 791 | 212 | 1,003 |

## Label Mapping Decision

This dataset is not vet-confirmed PetCare gold data. Treat it as Kaggle-derived
proxy training material only.

| Source class | PetCare mapping | Supervised use |
| --- | --- | --- |
| `ringworm` | `condition=dermatophytosis`, `oodClass=in_scope` | noisy proxy |
| `Fungal_infections` | `condition=fungal_malassezia`, `oodClass=in_scope` | noisy proxy |
| `Hypersensitivity` | `condition=atopic_dermatitis`, `oodClass=in_scope` | weak proxy |
| `Dermatitis` | `condition=unknown`, `oodClass=ambiguous_dermatitis_proxy` | no supervised canonical label |
| `Healthy` | `condition=unknown`, `oodClass=healthy_skin` | negative/OOD only |
| `demodicosis` | `condition=unknown`, `oodClass=unknown_ood` | OOD only |

## Commands

Prepare local manifest:

```bash
uv run python ml/prepare_hf_vet_derm.py \
  --source-dir ml/_raw/hf_vet_derm_dataset \
  --out ml/prepared/hf_vet_derm_dataset
```

Deduplicate against existing training/proxy candidates:

```bash
uv run --with pillow --with imagehash python ml/filter_roboflow_dedup.py \
  --candidate-manifest ml/prepared/hf_vet_derm_dataset/manifest.jsonl \
  --reference-manifest ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4/manifest.jsonl \
  --reference-manifest ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl \
  --reference-manifest ml/silver/silver-v0/manifest.jsonl \
  --reference-manifest ml/prepared/roboflow_dog_skin_disease_prediction_v3_dedup_phash4/manifest.jsonl \
  --out ml/prepared/hf_vet_derm_dataset_dedup_phash4_guarded \
  --filter-id hf_vet_derm_phash4_trainval
```

## Dedup Result

Reference rows loaded: 4,451
Candidate rows: 3,882
pHash threshold: 4

| Status | Rows |
| --- | ---: |
| Kept | 3,139 |
| Removed exact SHA matches | 374 |
| Removed pHash near-duplicates | 369 |
| Removed total | 743 |

Removed rows by reference manifest:

| Reference manifest | Removed rows |
| --- | ---: |
| `ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4/manifest.jsonl` | 666 |
| `ml/silver/silver-v0/manifest.jsonl` | 31 |
| `ml/prepared/roboflow_dog_skin_disease_prediction_v3_dedup_phash4/manifest.jsonl` | 30 |
| `ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl` | 16 |

Kept rows after dedup:

| Source class | Mapping | Kept rows |
| --- | --- | ---: |
| `Dermatitis` | ambiguous proxy, not supervised | 558 |
| `Fungal_infections` | `fungal_malassezia` noisy proxy | 324 |
| `Healthy` | `healthy_skin` negative/OOD | 403 |
| `Hypersensitivity` | `atopic_dermatitis` weak proxy | 240 |
| `demodicosis` | OOD only | 762 |
| `ringworm` | `dermatophytosis` noisy proxy | 852 |

## Decision

Do not use this dataset for evaluation and do not describe it as independent.
The duplicate rate and declared Kaggle lineage show that it overlaps materially
with our existing Roboflow/Kaggle-derived bootstrap.

Allowed use after dedup:

- noisy/proxy training expansion only;
- private HF publication only, if published;
- no clinical accuracy claims;
- no gold or silver evaluation membership;
- keep `Dermatitis` out of supervised canonical training until adjudicated.
