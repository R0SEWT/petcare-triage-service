# ml — YOLOv8-cls training (coarse baseline v0)

First dermatology-triage baseline. **Classification** (YOLOv8-cls), per
`../docs/ml/model-task-decision.md`. Trains on real Kaggle data
(`../docs/ml/data/dataset-scouting.md`) mapped to our canonical labels.

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
cd ml
pip install -r requirements.txt          # + CUDA torch matching the GPU
python prepare_data.py --download --out ./prepared --copy
python train_yolo_cls.py --data ./prepared --device 0 --epochs 50
```

`prepare_data.py --download` needs `~/.kaggle/kaggle.json` on the box.
Weights land in `runs/` (gitignored). Data/weights are never committed.

## Verify the mapping locally (no GPU)

```bash
python prepare_data.py --source-dir <downloaded>/Dogs --out ./prepared
```
Prints per-canonical-label train/val counts. Confirmed on
`yashmotiani/dogs-skin-disease-dataset` (CC0): 439 imgs → 4 canonical buckets.
