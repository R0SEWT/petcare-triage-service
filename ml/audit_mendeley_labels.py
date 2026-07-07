#!/usr/bin/env python3
"""Audit the incomplete Mendeley 5dbht54kw7 v1 label file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

PAPER_TOTALS = {
    "Bacterial_dermatosis": 23,
    "Fungal_infections": 19,
    "Hypersensitivity_allergic_dermatosis": 23,
    "Healthy": 30,
}

ADJUDICATION_COLUMNS = [
    "source_folder",
    "image_count",
    "current_source_label",
    "current_condition",
    "current_ood_class",
    "adjudicated_source_label",
    "adjudicated_condition",
    "adjudicated_ood_class",
    "reviewer_id",
    "reviewed_at",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("ml/_raw/mendeley_5dbht54kw7_v1"),
        help="Local raw Mendeley metadata directory.",
    )
    parser.add_argument(
        "--adjudication-csv",
        type=Path,
        help="Optional output CSV template for the unlabeled source folders.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_source_folders(folders_path: Path) -> dict[str, str]:
    rows = read_json(folders_path)
    folders: dict[str, str] = {}
    for row in rows:
        folder_id = row["id"]
        name = row["name"]
        if folder_id in folders:
            raise SystemExit(f"Duplicate folder id in {folders_path}: {folder_id}")
        folders[folder_id] = name
    return folders


def read_label_file(label_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            folder, label = line.split(None, 1)
        except ValueError as exc:
            raise SystemExit(f"Malformed label line {line_no}: {line!r}") from exc
        folder = folder.lower()
        if folder in labels:
            raise SystemExit(f"Duplicate folder label in {label_path}: {folder}")
        labels[folder] = label.strip()
    return labels


def image_counts_by_folder(files_path: Path, folders_by_id: dict[str, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in read_json(files_path):
        if row.get("filename") == "image_label.txt":
            continue
        folder_id = row.get("folder_id")
        if folder_id not in folders_by_id:
            raise SystemExit(f"File references unknown folder_id: {folder_id}")
        counts[folders_by_id[folder_id].lower()] += 1
    return counts


def missing_distribution(expected: dict[str, int], observed: Counter[str]) -> dict[str, int]:
    missing: dict[str, int] = {}
    for label, expected_count in expected.items():
        observed_count = observed.get(label, 0)
        if observed_count > expected_count:
            raise ValueError(
                f"Observed {observed_count} {label} labels, above expected total {expected_count}"
            )
        missing[label] = expected_count - observed_count
    return missing


def write_adjudication_csv(path: Path, missing_folders: list[str], image_counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADJUDICATION_COLUMNS)
        writer.writeheader()
        for folder in missing_folders:
            writer.writerow(
                {
                    "source_folder": folder,
                    "image_count": image_counts.get(folder.lower(), 0),
                    "current_source_label": "",
                    "current_condition": "unknown",
                    "current_ood_class": "unlabeled_source",
                    "adjudicated_source_label": "",
                    "adjudicated_condition": "",
                    "adjudicated_ood_class": "",
                    "reviewer_id": "",
                    "reviewed_at": "",
                    "notes": "",
                }
            )


def build_report(raw_dir: Path) -> dict[str, object]:
    folders_by_id = read_source_folders(raw_dir / "folders.json")
    labels_by_folder = read_label_file(raw_dir / "image_label.txt")
    image_counts = image_counts_by_folder(raw_dir / "files.json", folders_by_id)
    source_folders = {name.lower(): name for name in folders_by_id.values()}
    unknown_labeled = sorted(set(labels_by_folder) - set(source_folders))
    if unknown_labeled:
        raise SystemExit(f"Labels reference unknown folders: {', '.join(unknown_labeled)}")

    label_counts = Counter(labels_by_folder.values())
    missing_folders = sorted(
        source_folders[folder] for folder in set(source_folders) - set(labels_by_folder)
    )
    inferred_missing = missing_distribution(PAPER_TOTALS, label_counts)
    return {
        "source_folders": len(source_folders),
        "image_files": sum(image_counts.values()),
        "labeled_folders": len(labels_by_folder),
        "missing_folders": len(missing_folders),
        "label_file_counts": dict(sorted(label_counts.items())),
        "paper_reported_counts": PAPER_TOTALS,
        "inferred_missing_counts": inferred_missing,
        "missing_folder_names": missing_folders,
    }


def main() -> int:
    args = parse_args()
    report = build_report(args.raw_dir)
    if args.adjudication_csv:
        write_adjudication_csv(
            args.adjudication_csv,
            list(report["missing_folder_names"]),
            image_counts_by_folder(
                args.raw_dir / "files.json", read_source_folders(args.raw_dir / "folders.json")
            ),
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
