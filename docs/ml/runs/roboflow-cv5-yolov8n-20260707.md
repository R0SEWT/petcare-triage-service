# Roboflow cv5 YOLOv8n-cls bootstrap run

Task: `petcare-triage-service-owq`
Date: 2026-07-07
Remote: Lightning AI `ssh.lightning.ai`
Remote workspace: `/teamspace/studios/this_studio/petcare-triage-service`

## Data

Training source:

- `ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4_cv5`
- 5 group-aware cross-validation folds
- 4,198 filtered source rows
- 1,614 Roboflow augmentation groups
- no train/val group leakage within each fold

The dataset is noisy Roboflow bootstrap data. These metrics are not product
performance and must not be reported as independent accuracy.

## Command

```bash
python ml/train_yolo_cls.py \
  --folds-dir ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4_cv5 \
  --device 0 \
  --epochs 50 \
  --imgsz 224 \
  --batch 32 \
  --project /teamspace/studios/this_studio/petcare-triage-service/ml/runs/petcare-derm-yolov8-cls \
  --name cv5-yolov8n
```

Environment:

- GPU: Tesla T4, 15GB
- `torch`: 2.12.1+cu130
- `ultralytics`: 8.4.90
- base model: `yolov8n-cls.pt`

## Results

| Fold | Best epoch | Best top1 | Final top1 | Top5 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 21 | 0.93214 | 0.92738 | 1.00000 |
| 1 | 42 | 0.97262 | 0.96667 | 1.00000 |
| 2 | 41 | 0.96310 | 0.95238 | 1.00000 |
| 3 | 42 | 0.91061 | 0.90942 | 1.00000 |
| 4 | 36 | 0.92729 | 0.91895 | 1.00000 |

Aggregate over best fold checkpoints:

- mean top1: 0.941152
- stdev top1: 0.025875
- min top1: 0.91061
- max top1: 0.97262

## Artifacts

Remote and local gitignored copy:

- `ml/runs/petcare-derm-yolov8-cls/cv5-yolov8n-summary.json`
- `ml/runs/petcare-derm-yolov8-cls/logs/cv5-yolov8n-20260707-040639.log`
- `ml/runs/petcare-derm-yolov8-cls/cv5-yolov8n-fold_<n>/results.csv`
- `ml/runs/petcare-derm-yolov8-cls/cv5-yolov8n-fold_<n>/weights/best.pt`
- `ml/runs/petcare-derm-yolov8-cls/cv5-yolov8n-fold_<n>/weights/last.pt`

Local artifact size: 41M.

## Decision

This run is usable as a bootstrap model candidate and sanity check for the
training pipeline. Before any product use, evaluate on an independent
vet-verified frozen gold set, calibrate confidence thresholds, and preserve the
chosen checkpoint in a private artifact repository.
