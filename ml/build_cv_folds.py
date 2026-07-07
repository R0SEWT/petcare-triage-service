#!/usr/bin/env python3
"""Build stratified group cross-validation folds for YOLOv8 classification.

Roboflow exports contain multiple augmented variants of the same source image,
usually named like ``base_jpg.rf.<augmentation-id>.jpg``. This script assigns
those sibling variants as a group so they never appear in both train and val
within the same fold.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4/manifest.jsonl")
DEFAULT_OUT = Path("ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4_cv5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root for image_path values. Defaults to manifest parent.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite-links",
        action="store_true",
        help="Replace existing symlinks under fold train/val image layouts.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_manifest"] = str(path)
            row["_line"] = line_no
            rows.append(row)
    return rows


def label_for(row: dict[str, Any]) -> str:
    if row.get("oodClass") == "healthy_skin":
        return "healthy_skin"
    if row.get("oodClass") != "in_scope":
        raise SystemExit(
            f"Unsupported row for supervised CV at {row.get('_manifest')}:{row.get('_line')}: "
            f"oodClass={row.get('oodClass')}"
        )
    return str(row["condition"])


def augmentation_group_key(row: dict[str, Any]) -> str:
    source = Path(str(row.get("raw_path") or row["image_path"])).name
    stem = source.rsplit(".", 1)[0]
    base = stem.split(".rf.", 1)[0]
    return f"{row.get('source_class')}::{base}"


def safe_image_name(row: dict[str, Any]) -> str:
    raw_path = str(row.get("raw_path") or row["image_path"])
    return raw_path.replace("/", "__").replace(" ", "_")


def assign_groups(rows: list[dict[str, Any]], folds: int, seed: int) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[augmentation_group_key(row)].append(row)

    groups_by_label: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for group_key, group_rows in grouped.items():
        labels = {label_for(row) for row in group_rows}
        if len(labels) != 1:
            raise SystemExit(f"Group {group_key} spans multiple labels: {sorted(labels)}")
        groups_by_label[labels.pop()].append((group_key, group_rows))

    rng = random.Random(seed)
    fold_for_group: dict[str, int] = {}
    fold_label_counts: list[Counter[str]] = [Counter() for _ in range(folds)]
    fold_total_counts = [0 for _ in range(folds)]

    for label, label_groups in sorted(groups_by_label.items()):
        rng.shuffle(label_groups)
        label_groups.sort(key=lambda item: len(item[1]), reverse=True)
        for group_key, group_rows in label_groups:
            target_fold = min(
                range(folds),
                key=lambda fold: (fold_label_counts[fold][label], fold_total_counts[fold], fold),
            )
            fold_for_group[group_key] = target_fold
            fold_label_counts[target_fold][label] += len(group_rows)
            fold_total_counts[target_fold] += len(group_rows)

    return fold_for_group


def write_symlink(src: Path, dest: Path, overwrite_links: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if not overwrite_links:
            raise SystemExit(f"{dest} exists; pass --overwrite-links to refresh symlinks")
        if not dest.is_symlink():
            raise SystemExit(f"{dest} exists and is not a symlink; refusing to replace it")
        dest.unlink()
    dest.symlink_to(src.resolve())


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists; choose a new --out directory")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be >= 2")
    dataset_root = args.dataset_root or args.manifest.parent
    out_dir = args.out
    report_path = out_dir / "fold_report.json"
    if report_path.exists():
        raise SystemExit(f"{out_dir} already has fold outputs; choose a new --out directory")

    rows = load_jsonl(args.manifest)
    fold_for_group = assign_groups(rows, args.folds, args.seed)
    fold_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    report_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_counts: Counter[str] = Counter()

    for row in rows:
        label = label_for(row)
        group_key = augmentation_group_key(row)
        val_fold = fold_for_group[group_key]
        group_counts[label] += 1
        src = dataset_root / str(row["image_path"])
        if not src.is_file():
            raise SystemExit(f"Missing image for manifest row {row['_manifest']}:{row['_line']}: {src}")
        name = safe_image_name(row)

        for fold in range(args.folds):
            cv_split = "val" if fold == val_fold else "train"
            dest = out_dir / f"fold_{fold}" / cv_split / label / name
            write_symlink(src, dest, args.overwrite_links)
            record = {
                **clean_row(row),
                "cv_fold": fold,
                "cv_split": cv_split,
                "cv_label": label,
                "cv_group_key": group_key,
                "cv_group_val_fold": val_fold,
                "cv_folds": args.folds,
                "cv_seed": args.seed,
            }
            fold_rows[fold].append(record)
            report_counts[f"fold_{fold}:{cv_split}"][label] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        write_jsonl(out_dir / f"fold_{fold}" / "manifest.jsonl", fold_rows[fold])

    report = {
        "source_manifest": str(args.manifest),
        "dataset_root": str(dataset_root),
        "out_dir": str(out_dir),
        "folds": args.folds,
        "seed": args.seed,
        "source_rows": len(rows),
        "source_group_count": len(fold_for_group),
        "source_label_counts": dict(sorted(group_counts.items())),
        "fold_counts": {key: dict(sorted(counter.items())) for key, counter in sorted(report_counts.items())},
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {report_path}")
    print(f"source_rows {len(rows)}")
    print(f"source_groups {len(fold_for_group)}")
    for key, counter in sorted(report_counts.items()):
        total = sum(counter.values())
        labels = " ".join(f"{label}={count}" for label, count in sorted(counter.items()))
        print(f"{key} total={total} {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
