# ml — YOLOv8-cls training (coarse baseline v0)

First dermatology-triage baseline. **Classification** (YOLOv8-cls), per
`../docs/ml/model-task-decision.md`. Current bootstrap data comes from the
deduplicated Roboflow mirror documented in
`../docs/ml/data/roboflow-dog-skin-disease-v2.md`.

Runs on a **GPU box** (Lightning AI via SSH / Colab / HF Jobs) — not locally.
`prepare_data.py` is stdlib-only and verifiable without a GPU; training needs
`ultralytics` + `torch`. Dataset deduplication uses `pillow` + `imagehash`,
which are listed in `requirements.txt` and can also be injected locally with
`uv run --with pillow --with imagehash ...`.

## Label space (v0, coarse)

The dataset taxonomy is coarser than `../docs/ml/schema/labels.json`, so v0
lumps (see `label_map.json`):

| Model output (canonical id) | From dataset class | Coarse? |
| --- | --- | --- |
| `bacterial_pyoderma` | Bacterial_dermatosis | clean |
| `fungal_malassezia` | Fungal_infections | lumps dermatophytosis |
| `atopic_dermatitis` | Hypersensitivity_allergic_dermatosis | lumps allergic_contact |
| `healthy` | Healthy | OOD/negative |

Not predicted by v0 (no data): `dermatophytosis`, `allergic_contact_dermatitis`.
Output ids stay within the canonical enum, so the contract stays valid. The
vet-verified gold set (`../docs/ml/data/gold-eval-set.md`) is where the finer
5-class separation and any credible accuracy number come from — v0 numbers are
directional only.

## Run (on the Lightning AI GPU box)

```bash
ssh <lightning-box>
cd <petcare-triage-service>
pip install -r ml/requirements.txt       # + CUDA torch matching the GPU
python ml/build_cv_folds.py
python ml/train_yolo_cls.py --folds-dir ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4_cv5 --device 0 --epochs 50
```

Weights land in `runs/` (gitignored). Data/weights are never committed.
Use `--fold 0` to train only one fold for a smoke test before running all
folds.

## Verify the mapping locally (no GPU)

```bash
python ml/prepare_data.py --source-dir <downloaded>/Dogs --out ml/prepared
python ml/build_cv_folds.py --out "$(mktemp -d)/cv5_check"
```
`prepare_data.py` prints per-canonical-label train/val counts for legacy Kaggle
checks. `build_cv_folds.py` prints fold-level train/val counts and should report
balanced validation folds around 839-840 images each for the deduped Roboflow
bootstrap set.
