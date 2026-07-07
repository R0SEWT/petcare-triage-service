#!/usr/bin/env python3
"""Compare hosted Roboflow Universe models on a labeled local Roboflow export."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from roboflow_smoke_test import DEFAULT_MODELS, call_roboflow, parse_models

PRIMARY_LABEL_MAP = {
    "bacterial dermatosis": "bacterial dermatosis",
    "fungal infection": "fungal infection",
    "healthy": "healthy",
    "hypersensitivity dermatitis": "hypersensitivity dermatitis",
}

CLINICAL_LABEL_MAP = {
    "bacterial dermatosis": None,
    "fungal infection": "ringworm",
    "healthy": None,
    "hypersensitivity dermatitis": "flea_allergy",
}

DETECTOR_LABEL_MAP = {
    "bacterial dermatosis": "bacterial-dermatosis",
    "fungal infection": "fungal-infection",
    "healthy": "healthy",
    "hypersensitivity dermatitis": "hypersensitivity-allergic-dermatosis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ml/prepared/roboflow_dog_skin_disease_dataset_v2/manifest.jsonl"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("ml/prepared/roboflow_dog_skin_disease_dataset_v2"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ml/runs/roboflow_model_compare"),
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", action="append", metavar="TASK:MODEL_ID")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_rows(rows: list[dict[str, Any]], split: str, per_class: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[str(row["source_class"])].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for source_class in sorted(grouped):
        items = grouped[source_class]
        rng.shuffle(items)
        selected.extend(items[:per_class])
    return selected


def prediction_for(model_id: str, response: dict[str, Any]) -> tuple[str | None, float | None]:
    if model_id == "dog-skin-disease-dataset/2":
        pred = response.get("top")
        conf = response.get("confidence")
        return pred, conf
    if model_id == "dog-skin-disease-prediction/3":
        preds = response.get("predicted_classes") or []
        scores = response.get("predictions") or {}
        if preds:
            pred = preds[0]
            return pred, scores.get(pred, {}).get("confidence")
        if scores:
            pred = max(scores, key=lambda key: scores[key].get("confidence", 0))
            return pred, scores[pred].get("confidence")
        return None, None
    if model_id == "dog-skin-diseases/1":
        detections = response.get("predictions") or []
        if not detections:
            return "no_detection", 0.0
        best = max(detections, key=lambda item: item.get("confidence", 0))
        return best.get("class"), best.get("confidence")

    predictions = response.get("predictions")
    if isinstance(predictions, list) and predictions:
        best = max(predictions, key=lambda item: item.get("confidence", 0))
        return best.get("class"), best.get("confidence")
    return response.get("top"), response.get("confidence")


def expected_for(model_id: str, source_class: str) -> str | None:
    if model_id == "dog-skin-disease-dataset/2":
        return PRIMARY_LABEL_MAP[source_class]
    if model_id == "dog-skin-disease-prediction/3":
        return CLINICAL_LABEL_MAP[source_class]
    if model_id == "dog-skin-diseases/1":
        return DETECTOR_LABEL_MAP[source_class]
    return source_class


def main() -> int:
    args = parse_args()
    rows = sample_rows(load_rows(args.manifest), args.split, args.per_class, args.seed)
    models = parse_models(args.model) if args.model else DEFAULT_MODELS

    print(f"selected={len(rows)} split={args.split} per_class={args.per_class}")
    class_counts = Counter(row["source_class"] for row in rows)
    for source_class, count in sorted(class_counts.items()):
        print(f"class {source_class}: {count}")

    if args.dry_run:
        return 0

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set; run with --dry-run or export the key.", file=sys.stderr)
        return 2

    run_dir = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "summary.json"

    stats: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_path = args.dataset_root / row["image_path"]
            for model in models:
                result = call_roboflow(model, image_path, api_key, args.timeout)
                pred = None
                conf = None
                expected = expected_for(model["model_id"], str(row["source_class"]))
                comparable = expected is not None
                correct = None
                if result["ok"]:
                    pred, conf = prediction_for(model["model_id"], result["response"])
                    if comparable:
                        correct = pred == expected
                stats[model["model_id"]]["total"] += 1
                stats[model["model_id"]]["ok" if result["ok"] else "fail"] += 1
                if comparable:
                    stats[model["model_id"]]["comparable"] += 1
                    stats[model["model_id"]]["correct" if correct else "wrong"] += 1
                    confusion[model["model_id"]][f"{row['source_class']}->{pred}"] += 1
                record = {
                    "model": model,
                    "source_class": row["source_class"],
                    "expected": expected,
                    "comparable": comparable,
                    "prediction": pred,
                    "confidence": conf,
                    "correct": correct,
                    "row": row,
                    "result": result,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    "result",
                    model["model_id"],
                    row["source_class"],
                    "->",
                    pred,
                    "expected",
                    expected,
                    "ok" if result["ok"] else "fail",
                    f"{result['latency_ms']}ms",
                )

    summary = {
        "split": args.split,
        "per_class": args.per_class,
        "seed": args.seed,
        "rows": len(rows),
        "models": models,
        "stats": {model: dict(counter) for model, counter in stats.items()},
        "confusion": {model: dict(counter) for model, counter in confusion.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote={predictions_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
