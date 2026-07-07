# Roboflow Dog Skin Disease Dataset v2 acquisition

Task: `petcare-triage-service-jqj`
Source: https://universe.roboflow.com/dog-skin-disease-dermatosis/dog-skin-disease-dataset
Roboflow workspace/project/version: `dog-skin-disease-dermatosis/dog-skin-disease-dataset/2`
Export format: `folder`
Status: acquired locally as noisy bootstrap data, not approved for evaluation.

## Local layout

Raw export:

- `ml/_raw/roboflow_probe_folder/`
- `ml/_raw/roboflow_probe_folder/README.dataset.txt`
- `ml/_raw/roboflow_probe_folder/README.roboflow.txt`
- `ml/_raw/roboflow_probe_folder/{train,valid,test}/<source-class>/*.jpg`

Prepared local layout:

- `ml/prepared/roboflow_dog_skin_disease_dataset_v2/manifest.jsonl`
- `ml/prepared/roboflow_dog_skin_disease_dataset_v2/images/<split>/<bucket>/*.jpg`

Both raw and prepared paths are gitignored.

## Source metadata

Uploader-declared license: CC BY 4.0.

Roboflow README says:

- 4,398 images
- folder-format classification export
- auto-orientation with EXIF stripping
- resized to `640x640` using stretch
- augmentation creates 3 versions of each source image
- augmentation includes brightness +/-20% and exposure +/-10%

Local validation:

- images: 4,398
- files including README files: 4,400
- disk usage: 198M
- image geometry/channels: all `640x640`, `sRGB`, 3 channels

## Class counts

| Split | Source class | Count | PetCare mapping |
| --- | --- | ---: | --- |
| train | `bacterial dermatosis` | 954 | `condition=bacterial_pyoderma`, `oodClass=in_scope` |
| train | `fungal infection` | 1,053 | `condition=fungal_malassezia`, `oodClass=in_scope` |
| train | `healthy` | 1,037 | `condition=unknown`, `oodClass=healthy_skin` |
| train | `hypersensitivity dermatitis` | 807 | `condition=atopic_dermatitis`, `oodClass=in_scope` |
| val | `bacterial dermatosis` | 98 | `condition=bacterial_pyoderma`, `oodClass=in_scope` |
| val | `fungal infection` | 101 | `condition=fungal_malassezia`, `oodClass=in_scope` |
| val | `healthy` | 98 | `condition=unknown`, `oodClass=healthy_skin` |
| val | `hypersensitivity dermatitis` | 70 | `condition=atopic_dermatitis`, `oodClass=in_scope` |
| test | `bacterial dermatosis` | 45 | `condition=bacterial_pyoderma`, `oodClass=in_scope` |
| test | `fungal infection` | 47 | `condition=fungal_malassezia`, `oodClass=in_scope` |
| test | `healthy` | 51 | `condition=unknown`, `oodClass=healthy_skin` |
| test | `hypersensitivity dermatitis` | 37 | `condition=atopic_dermatitis`, `oodClass=in_scope` |

No `Unlabeled` folder was present in this `folder` export even though the
Roboflow project metadata lists an `Unlabeled` class.

## Deduplication against Mendeley

Compared Roboflow v2 against `ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl`.

Exact SHA-256:

- Mendeley hashes: 99
- Roboflow hashes: 4,397 unique among 4,398 images
- exact overlap: 0

Perceptual hash nearest-neighbor:

- nearest pHash distance <= 0: 6 Mendeley images
- nearest pHash distance <= 4: 8 Mendeley images
- nearest pHash distance <= 8: 8 Mendeley images

Examples at distance <= 4 include Mendeley healthy images matching Roboflow
healthy images and one Mendeley `atopic_dermatitis` image matching Roboflow
`hypersensitivity dermatitis`.

Decision: Roboflow likely contains resized/augmented derivatives from the same
lineage as Mendeley. Before training, create a filtered manifest that excludes
Roboflow images whose pHash is near Mendeley or any future gold-eval image. Do
not use this Roboflow test split as independent evaluation.

## Hosted model comparison

Comparison sample:

- local output: `ml/runs/roboflow_model_compare/20260706-201152/`
- split: Roboflow `test`
- sample: 10 images per class, 40 images total
- requests: 120, across 3 hosted models
- result: 120/120 HTTP 200

| Model | Comparable labels | Correct | Accuracy on comparable sample | Notes |
| --- | ---: | ---: | ---: | --- |
| `dog-skin-disease-dataset/2` | 40 | 39 | 97.5% | Strong on its own Roboflow v2 test sample |
| `dog-skin-disease-prediction/3` | 20 | 15 | 75.0% | Only mapped fungal->ringworm and hypersensitivity->flea_allergy; bacterial/healthy are taxonomy-incompatible |
| `dog-skin-diseases/1` | 40 | 5 | 12.5% | Detection model is not reliable as a classifier |

Important contrast: `dog-skin-disease-dataset/2` performed well on its own
Roboflow export, but over-predicted `healthy` on Mendeley probes during the
earlier smoke test. Treat this as evidence of dataset/domain leakage or narrow
domain fit, not as PetCare-ready performance.

## Training decision

Use Roboflow v2 only as noisy bootstrap training data for a PetCare-owned model.
The next training dataset should:

- exclude near-duplicates against Mendeley and any future gold set
- keep `healthy` as `oodClass=healthy_skin`, not a condition label
- treat `fungal infection` -> `fungal_malassezia` as approximate
- never report Roboflow held-out accuracy as product performance
- evaluate only on an independent, vet-verified frozen gold set

## Verification commands

```bash
find ml/_raw/roboflow_probe_folder -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
find ml/_raw/roboflow_probe_folder -mindepth 2 -maxdepth 3 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | awk -F/ '{print $(NF-2)"/"$(NF-1)}' | sort | uniq -c
find ml/_raw/roboflow_probe_folder -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0 | xargs -0 identify -format '%w %h %[colorspace] %[channels]\n' | sort | uniq -c
uv run python ml/prepare_roboflow.py
uv run python ml/roboflow_compare_models.py --per-class 10
```
