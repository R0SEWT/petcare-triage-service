"""Train the coarse-v0 YOLOv8 classification baseline.

Runs on a GPU box (Lightning AI / Colab / HF Jobs). Not run locally — needs
ultralytics + torch. Prepare the data first with prepare_data.py.

Usage (on the GPU box):
    pip install -r ml/requirements.txt
    python ml/prepare_data.py --download --out ml/prepared --copy
    python ml/train_yolo_cls.py --data ml/prepared --device 0 --epochs 50
    python ml/build_cv_folds.py
    python ml/train_yolo_cls.py --folds-dir ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4_cv5 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train_one(args: argparse.Namespace, data: str, name: str) -> None:
    from ultralytics import YOLO  # imported here so the file is inspectable without torch

    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=name,
    )
    metrics = model.val()
    print(name, "top1:", getattr(metrics, "top1", None), "top5:", getattr(metrics, "top5", None))


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Prepared dataset dir (has train/ and val/)")
    source.add_argument("--folds-dir", type=Path, help="Crossval directory containing fold_<n>/ dirs")
    ap.add_argument(
        "--fold",
        type=int,
        action="append",
        help="Fold index to train. Repeatable. Defaults to all fold_* dirs when --folds-dir is used.",
    )
    ap.add_argument("--model", default="yolov8n-cls.pt", help="yolov8n/s/m-cls.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    ap.add_argument("--project", default="runs/petcare-derm-yolov8-cls")
    ap.add_argument("--name", default="v0-coarse", help="Run name prefix")
    args = ap.parse_args()

    if args.data:
        train_one(args, args.data, args.name)
    else:
        folds_dir = args.folds_dir
        folds = sorted(p for p in folds_dir.iterdir() if p.is_dir() and p.name.startswith("fold_"))
        if args.fold is not None:
            wanted = {f"fold_{fold}" for fold in args.fold}
            folds = [path for path in folds if path.name in wanted]
            missing = wanted - {path.name for path in folds}
            if missing:
                raise SystemExit(f"Missing requested folds under {folds_dir}: {sorted(missing)}")
        if not folds:
            raise SystemExit(f"No fold_* directories found under {folds_dir}")
        for fold_dir in folds:
            train_one(args, str(fold_dir), f"{args.name}-{fold_dir.name}")
    print("Best weights under:", args.project)


if __name__ == "__main__":
    main()
