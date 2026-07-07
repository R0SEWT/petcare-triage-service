#!/usr/bin/env python3
"""Create a silver proxy intake CSV from a prepared dataset manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

FIELDNAMES = [
    "source_path",
    "image_id",
    "source",
    "license",
    "species",
    "breed",
    "body_region",
    "condition",
    "ood_class",
    "quality_flags",
    "vet_confirmed",
    "adjudication_mode",
    "annotator_id",
    "labeled_at",
    "labelers",
    "agreed",
    "tiebreak",
    "consent_scope",
    "notes",
    "storage_ref",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("ml/prepared/roboflow_dog_skin_disease_dataset_v2/manifest.jsonl"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Root for source manifest image_path values. Defaults to source manifest parent.",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--split", default="test", help="Source split to sample.")
    parser.add_argument("--max-per-bucket", type=int, default=20, help="0 means no cap.")
    parser.add_argument("--id-prefix", default="silver_roboflow_test")
    parser.add_argument("--adjudication-mode", default="simulated")
    parser.add_argument("--annotator-id", default="proxy_roboflow")
    parser.add_argument("--labeler", action="append", help="Repeatable labeler id. Defaults to annotator id.")
    parser.add_argument("--labeled-at", default=date.today().isoformat())
    parser.add_argument("--consent-scope", default="eval_only")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path. Defaults to <out-csv>.report.json.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line"] = line_no
            rows.append(row)
    return rows


def bucket_key(row: dict[str, Any]) -> str:
    ood_class = str(row.get("oodClass") or "unknown_ood")
    if ood_class == "in_scope":
        return str(row.get("condition") or "unknown")
    return ood_class


def source_path_for_csv(out_csv: Path, source_root: Path, image_path: str) -> str:
    absolute = (source_root / image_path).resolve()
    return os.path.relpath(absolute, start=out_csv.parent.resolve())


def intake_rows(
    manifest_rows: list[dict[str, Any]],
    *,
    source_root: Path,
    out_csv: Path,
    split: str,
    max_per_bucket: int,
    id_prefix: str,
    adjudication_mode: str,
    annotator_id: str,
    labelers: list[str],
    labeled_at: str,
    consent_scope: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    skipped_missing_image = 0
    candidates = sorted(
        (row for row in manifest_rows if str(row.get("split")) == split),
        key=lambda row: (bucket_key(row), str(row.get("image_path") or "")),
    )

    for row in candidates:
        bucket = bucket_key(row)
        if max_per_bucket and counts[bucket] >= max_per_bucket:
            continue
        image_path = str(row.get("image_path") or "")
        if not (source_root / image_path).is_file():
            skipped_missing_image += 1
            continue
        counts[bucket] += 1
        selected.append(
            {
                "source_path": source_path_for_csv(out_csv, source_root, image_path),
                "image_id": f"{id_prefix}_{len(selected) + 1:06d}",
                "source": str(row.get("source_url") or row.get("dataset_id") or row.get("source") or "unknown"),
                "license": str(row.get("license") or "unknown"),
                "species": str(row.get("species") or "unknown"),
                "breed": str(row.get("breed") or "unknown"),
                "body_region": str(row.get("bodyRegion") or row.get("body_region") or "unknown"),
                "condition": str(row.get("condition") or "unknown"),
                "ood_class": str(row.get("oodClass") or "unknown_ood"),
                "quality_flags": "",
                "vet_confirmed": "false",
                "adjudication_mode": adjudication_mode,
                "annotator_id": annotator_id,
                "labeled_at": labeled_at,
                "labelers": ";".join(labelers),
                "agreed": "true",
                "tiebreak": "",
                "consent_scope": consent_scope,
                "notes": (
                    "Proxy silver row sampled from held-out prepared manifest split; "
                    "not vet-confirmed and not valid for clinical accuracy claims."
                ),
                "storage_ref": "",
            }
        )

    report = {
        "sourceRows": len(manifest_rows),
        "candidateRows": len(candidates),
        "selectedRows": len(selected),
        "split": split,
        "maxPerBucket": max_per_bucket,
        "countsByBucket": dict(sorted(counts.items())),
        "skippedMissingImage": skipped_missing_image,
    }
    return selected, report


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} already exists; choose a new --out-csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    source_root = args.source_root or args.source_manifest.parent
    labelers = args.labeler or [args.annotator_id]
    rows, report = intake_rows(
        load_jsonl(args.source_manifest),
        source_root=source_root,
        out_csv=args.out_csv,
        split=args.split,
        max_per_bucket=args.max_per_bucket,
        id_prefix=args.id_prefix,
        adjudication_mode=args.adjudication_mode,
        annotator_id=args.annotator_id,
        labelers=labelers,
        labeled_at=args.labeled_at,
        consent_scope=args.consent_scope,
    )
    if not rows:
        raise SystemExit("No silver intake rows selected")

    write_csv(args.out_csv, rows)
    report_path = args.report or args.out_csv.with_suffix(args.out_csv.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
