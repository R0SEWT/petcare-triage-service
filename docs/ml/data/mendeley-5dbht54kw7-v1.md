# Mendeley 5dbht54kw7 v1 acquisition notes

Task: `petcare-triage-service-e7k`
Source: https://data.mendeley.com/datasets/5dbht54kw7/1
DOI: `10.17632/5dbht54kw7.1`
License: CC BY 4.0
Published: 2022-02-04

## Source summary

Dataset title: "Classification of pet dog skin diseases using deep learning
with images captured from multispectral imaging device".

The Mendeley record describes images from 95 pet dogs, collected after written
owner consent, with reported dog-level counts:

- bacterial dermatosis: 23
- fungal infections: 19
- hypersensitivity allergic dermatosis: 23
- healthy: 30

Downloaded local raw layout:

- `ml/_raw/mendeley_5dbht54kw7_v1/files.json`
- `ml/_raw/mendeley_5dbht54kw7_v1/folders.json`
- `ml/_raw/mendeley_5dbht54kw7_v1/image_label.txt`
- `ml/_raw/mendeley_5dbht54kw7_v1/images/<source-folder>/pic*.jpg`

Local normalized layout:

- `ml/prepared/mendeley_5dbht54kw7_v1/images/<bucket>/<source-folder>__pic*.jpg`
- `ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl`

Both raw and prepared layouts are gitignored.

## Verification

Downloaded file inventory from Mendeley public metadata:

- 99 JPEG files
- 127,910,054 JPEG bytes
- 0 SHA-256 mismatches against `files.json`
- all images: `1920x1080`, `srgb`, 3 channels

Verification commands:

```bash
find ml/_raw/mendeley_5dbht54kw7_v1/images -type f -name '*.jpg' | wc -l
find ml/_raw/mendeley_5dbht54kw7_v1/images -type f -name '*.jpg' -printf '%s\n' | awk '{s+=$1} END {print s}'
find ml/_raw/mendeley_5dbht54kw7_v1/images -type f -name '*.jpg' -print0 | xargs -0 identify -format '%w %h %[channels]\n' | sort | uniq -c
```

## Label status

Important discrepancy: `folders.json` contains 95 source folders, but
`image_label.txt` contains labels for only 62 folders. The missing 33 folders
are preserved as `unlabeled_source` and must not be used as supervised condition
examples until a complete label table is found or labels are adjudicated.

Image-level normalized counts from the available label file:

| Source label | PetCare mapping | Image count |
| --- | --- | ---: |
| `Bacterial_dermatosis` | `condition=bacterial_pyoderma`, `oodClass=in_scope` | 13 |
| `Fungal_infections` | `condition=fungal_malassezia`, `oodClass=in_scope` | 11 |
| `Hypersensitivity_allergic_dermatosis` | `condition=atopic_dermatitis`, `oodClass=in_scope` | 13 |
| `Healthy` | `condition=unknown`, `oodClass=healthy_skin` | 29 |
| missing from `image_label.txt` | `condition=unknown`, `oodClass=unlabeled_source` | 33 |

Training use is acceptable only for a noisy coarse baseline. Evaluation use is
not acceptable unless labels are vet-confirmed and deduplicated against the
frozen gold set.
