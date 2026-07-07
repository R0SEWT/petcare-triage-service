#!/usr/bin/env python3
"""Audit silver-v0 proxy intake coverage before publishing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.audit_gold_intake import audit_rows
from ml.build_silver_manifest import read_intake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-csv", type=Path, required=True, help="Proxy adjudicated intake CSV.")
    parser.add_argument("--silver-root", type=Path, default=Path("ml/silver/silver-v0"))
    parser.add_argument("--dataset-version", default="silver-v0")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("docs/ml/schema/silver-manifest.schema.json"),
        help="Silver manifest row JSON Schema.",
    )
    parser.add_argument("--target-per-bucket", type=int, default=20)
    parser.add_argument("--fail-on-deficit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_rows = [
        row for row, _, _ in read_intake(args.intake_csv, args.silver_root, args.dataset_version, args.schema)
    ]
    summary = audit_rows(manifest_rows, args.target_per_bucket)
    summary["validationTier"] = "proxy"
    summary["datasetVersion"] = args.dataset_version
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_deficit and not summary["complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
