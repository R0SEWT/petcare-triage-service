#!/usr/bin/env python3
"""Audit gold-v0 intake coverage before freezing the manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ml.build_gold_manifest import read_intake

IN_SCOPE_CONDITIONS = [
    "atopic_dermatitis",
    "dermatophytosis",
    "allergic_contact_dermatitis",
    "fungal_malassezia",
    "bacterial_pyoderma",
]

OOD_BUCKETS = [
    "healthy_skin",
    "non_skin_pet",
    "human_skin",
    "other_species",
    "poor_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-csv", type=Path, required=True, help="Adjudicated intake CSV.")
    parser.add_argument("--gold-root", type=Path, default=Path("ml/gold/gold-v0"))
    parser.add_argument("--dataset-version", default="gold-v0")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("docs/ml/schema/gold-manifest.schema.json"),
        help="Gold manifest row JSON Schema.",
    )
    parser.add_argument(
        "--target-per-bucket",
        type=int,
        default=20,
        help="Minimum rows per v0 condition/OOD bucket.",
    )
    parser.add_argument(
        "--fail-on-deficit",
        action="store_true",
        help="Exit non-zero when any v0 target bucket is under target.",
    )
    return parser.parse_args()


def audit_rows(rows: list[dict[str, Any]], target_per_bucket: int) -> dict[str, Any]:
    condition_counts: Counter[str] = Counter()
    ood_counts: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()

    for row in rows:
        ood_class = str(row["oodClass"])
        species_counts[str(row["species"])] += 1
        for flag in row["qualityFlags"]:
            quality_counts[str(flag)] += 1
        if ood_class == "in_scope":
            condition_counts[str(row["condition"])] += 1
        else:
            ood_counts[ood_class] += 1

    targets = {
        "conditions": {condition: target_per_bucket for condition in IN_SCOPE_CONDITIONS},
        "ood": {bucket: target_per_bucket for bucket in OOD_BUCKETS},
    }
    deficits = {
        "conditions": {
            condition: max(0, target - condition_counts[condition])
            for condition, target in targets["conditions"].items()
        },
        "ood": {bucket: max(0, target - ood_counts[bucket]) for bucket, target in targets["ood"].items()},
    }

    complete = all(value == 0 for group in deficits.values() for value in group.values())
    return {
        "rows": len(rows),
        "targetPerBucket": target_per_bucket,
        "complete": complete,
        "counts": {
            "conditions": dict(sorted(condition_counts.items())),
            "ood": dict(sorted(ood_counts.items())),
            "species": dict(sorted(species_counts.items())),
            "qualityFlags": dict(sorted(quality_counts.items())),
        },
        "targets": targets,
        "deficits": deficits,
    }


def main() -> int:
    args = parse_args()
    manifest_rows = [
        row for row, _, _ in read_intake(args.intake_csv, args.gold_root, args.dataset_version, args.schema)
    ]
    summary = audit_rows(manifest_rows, args.target_per_bucket)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_deficit and not summary["complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
