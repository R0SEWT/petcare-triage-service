"""Train the coarse-v0 YOLOv8 classification baseline.

Runs on a GPU box (Lightning AI / Colab / HF Jobs). Not run locally — needs
ultralytics + torch. Prepare the data first with prepare_data.py.

Usage (on the GPU box):
    pip install -r requirements.txt
    python prepare_data.py --download --out ./prepared --copy
    python train_yolo_cls.py --data ./prepared --device 0 --epochs 50
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Prepared dataset dir (has train/ and val/)")
    ap.add_argument("--model", default="yolov8n-cls.pt", help="yolov8n/s/m-cls.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    ap.add_argument("--project", default="runs/petcare-derm-yolov8-cls")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so the file is inspectable without torch

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="v0-coarse",
    )
    metrics = model.val()
    print("top1:", getattr(metrics, "top1", None), "top5:", getattr(metrics, "top5", None))
    print("Best weights under:", args.project)


if __name__ == "__main__":
    main()
