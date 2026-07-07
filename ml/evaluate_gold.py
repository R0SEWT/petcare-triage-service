#!/usr/bin/env python3
"""Evaluate PetCare classifier checkpoints against a frozen gold manifest.

The harness intentionally validates the gold manifest before inference and
keeps OOD/quality rows separate from in-scope condition accuracy. Current
YOLOv8-cls bootstrap checkpoints do not implement the future OOD gate; they can
only be assessed on in-scope condition rows and healthy_skin negatives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

IN_SCOPE_LABELS = {
    "atopic_dermatitis",
    "dermatophytosis",
    "allergic_contact_dermatitis",
    "fungal_malassezia",
    "bacterial_pyoderma",
}
NEGATIVE_LABEL = "healthy_skin"
ECE_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen gold manifest JSONL.")
    parser.add_argument("--image-root", type=Path, required=True, help="Root for manifest imagePath values.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("docs/ml/schema/gold-manifest.schema.json"),
        help="Gold manifest row JSON Schema.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="YOLOv8-cls checkpoint. Repeat for multiple folds/checkpoints.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("ml/runs/gold_eval"))
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="If max confidence is below this threshold, emit unknown.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading YOLO.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line"] = line_no
            image_id = str(row.get("imageId"))
            digest = str(row.get("sha256"))
            if image_id in seen_ids:
                raise SystemExit(f"Duplicate imageId at line {line_no}: {image_id}")
            if digest in seen_hashes:
                raise SystemExit(f"Duplicate sha256 at line {line_no}: {digest}")
            seen_ids.add(image_id)
            seen_hashes.add(digest)
            rows.append(row)
    if not rows:
        raise SystemExit(f"Gold manifest is empty: {path}")
    return rows


def validate_rows(rows: list[dict[str, Any]], image_root: Path, schema_path: Path) -> dict[str, Any]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    counts: Counter[str] = Counter()
    missing: list[str] = []
    sha_mismatch: list[str] = []
    for row in rows:
        errors = sorted(validator.iter_errors({k: v for k, v in row.items() if not k.startswith("_")}), key=str)
        if errors:
            first = errors[0]
            path = ".".join(str(part) for part in first.path) or "<row>"
            raise SystemExit(f"Schema error at line {row['_line']} path {path}: {first.message}")
        if row.get("vetConfirmed") is not True:
            raise SystemExit(f"Gold row must be vetConfirmed=true at line {row['_line']}")
        if row.get("neverTrain") is not True:
            raise SystemExit(f"Gold row must be neverTrain=true at line {row['_line']}")
        ood_class = str(row["oodClass"])
        condition = str(row["condition"])
        if ood_class == "in_scope" and condition not in IN_SCOPE_LABELS:
            raise SystemExit(f"Invalid in-scope condition at line {row['_line']}: {condition}")
        if ood_class != "in_scope" and condition != "unknown":
            raise SystemExit(f"OOD/negative rows must use condition=unknown at line {row['_line']}")

        image_path = image_root / str(row["imagePath"])
        if not image_path.is_file():
            missing.append(str(image_path))
        elif sha256(image_path) != row["sha256"]:
            sha_mismatch.append(str(image_path))
        counts[f"oodClass:{ood_class}"] += 1
        counts[f"condition:{condition}"] += 1

    if missing:
        raise SystemExit(f"Missing {len(missing)} gold images. First missing: {missing[0]}")
    if sha_mismatch:
        raise SystemExit(f"SHA mismatch for {len(sha_mismatch)} gold images. First mismatch: {sha_mismatch[0]}")
    return {"rows": len(rows), "counts": dict(sorted(counts.items()))}


def yolo_prediction(model: Any, image_path: Path, imgsz: int, threshold: float) -> dict[str, Any]:
    result = model.predict(str(image_path), imgsz=imgsz, verbose=False)[0]
    names = result.names
    probs = result.probs
    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf)
    top5 = []
    for idx in probs.top5:
        top5.append({"label": names[int(idx)], "confidence": float(probs.data[int(idx)])})
    label = names[top1_idx]
    if top1_conf < threshold:
        label = "unknown"
    return {"prediction": label, "confidence": top1_conf, "top5": top5}


def calibration_ece(records: list[dict[str, Any]], bins: int = ECE_BINS) -> float | None:
    scored = [record for record in records if record["scored"]]
    if not scored:
        return None
    total = len(scored)
    ece = 0.0
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        bucket = [
            record
            for record in scored
            if lower <= float(record["confidence"]) < upper or (idx == bins - 1 and float(record["confidence"]) == 1.0)
        ]
        if not bucket:
            continue
        accuracy = sum(1 for record in bucket if record["correct"]) / len(bucket)
        confidence = sum(float(record["confidence"]) for record in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(accuracy - confidence)
    return ece


def per_label_metrics(confusion: dict[str, Counter[str]]) -> dict[str, dict[str, float | int | None]]:
    labels = sorted(set(confusion) | {prediction for counter in confusion.values() for prediction in counter})
    metrics: dict[str, dict[str, float | int | None]] = {}
    for label in labels:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(counter[label] for counter in confusion.values())
        precision = true_positive / predicted if predicted else None
        recall = true_positive / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        metrics[label] = {
            "support": support,
            "predicted": predicted,
            "truePositive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def evaluate_checkpoint(
    checkpoint: Path,
    rows: list[dict[str, Any]],
    image_root: Path,
    out_dir: Path,
    imgsz: int,
    threshold: float,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    checkpoint_id = checkpoint.parent.parent.name if checkpoint.name in {"best.pt", "last.pt"} else checkpoint.stem
    records_path = out_dir / f"{checkpoint_id}.predictions.jsonl"
    rows_out: list[dict[str, Any]] = []
    metrics = {
        "checkpoint": str(checkpoint),
        "checkpointId": checkpoint_id,
        "confidenceThreshold": threshold,
        "totalRows": len(rows),
        "conditionRows": 0,
        "conditionCorrect": 0,
        "healthyRows": 0,
        "healthyCorrect": 0,
        "oodRowsNotScored": 0,
    }
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_path = image_root / str(row["imagePath"])
            pred = yolo_prediction(model, image_path, imgsz, threshold)
            expected = str(row["condition"])
            ood_class = str(row["oodClass"])
            scored = False
            correct: bool | None = None
            if ood_class == "in_scope":
                scored = True
                metrics["conditionRows"] += 1
                correct = pred["prediction"] == expected
                metrics["conditionCorrect"] += int(correct)
                confusion[expected][pred["prediction"]] += 1
            elif ood_class == "healthy_skin":
                scored = True
                metrics["healthyRows"] += 1
                correct = pred["prediction"] == NEGATIVE_LABEL
                metrics["healthyCorrect"] += int(correct)
                confusion[NEGATIVE_LABEL][pred["prediction"]] += 1
            else:
                metrics["oodRowsNotScored"] += 1

            record = {
                "imageId": row["imageId"],
                "imagePath": row["imagePath"],
                "expectedCondition": expected,
                "oodClass": ood_class,
                "scored": scored,
                "correct": correct,
                **pred,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            rows_out.append(record)

    metrics["conditionAccuracy"] = (
        metrics["conditionCorrect"] / metrics["conditionRows"] if metrics["conditionRows"] else None
    )
    metrics["healthyAccuracy"] = metrics["healthyCorrect"] / metrics["healthyRows"] if metrics["healthyRows"] else None
    scored_rows = metrics["conditionRows"] + metrics["healthyRows"]
    scored_correct = metrics["conditionCorrect"] + metrics["healthyCorrect"]
    metrics["scoredAccuracy"] = scored_correct / scored_rows if scored_rows else None
    metrics["confusion"] = {label: dict(counter) for label, counter in sorted(confusion.items())}
    metrics["perLabel"] = per_label_metrics(confusion)
    metrics["abstentionRows"] = sum(1 for record in rows_out if record["scored"] and record["prediction"] == "unknown")
    metrics["abstentionCoverage"] = metrics["abstentionRows"] / scored_rows if scored_rows else None
    metrics["ece"] = calibration_ece(rows_out)

    csv_path = out_dir / f"{checkpoint_id}.predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "imageId",
                "imagePath",
                "expectedCondition",
                "oodClass",
                "prediction",
                "confidence",
                "scored",
                "correct",
            ],
        )
        writer.writeheader()
        for record in rows_out:
            writer.writerow({key: record[key] for key in writer.fieldnames})

    return metrics


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest)
    validation = validate_rows(rows, args.image_root, args.schema)
    out_dir = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "manifest": str(args.manifest),
        "schema": str(args.schema),
        "imageRoot": str(args.image_root),
        "validation": validation,
        "imgsz": args.imgsz,
        "confidenceThreshold": args.confidence_threshold,
        "checkpoints": [str(path) for path in args.checkpoint],
        "dryRun": args.dry_run,
        "results": [],
    }
    if not args.dry_run:
        for checkpoint in args.checkpoint:
            summary["results"].append(
                evaluate_checkpoint(checkpoint, rows, args.image_root, out_dir, args.imgsz, args.confidence_threshold)
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
