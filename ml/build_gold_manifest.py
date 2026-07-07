#!/usr/bin/env python3
"""Build a frozen gold manifest from an adjudicated intake CSV.

The CSV is intentionally explicit: it must carry source, license, consent, vet
confirmation, and adjudication fields before a row can enter the frozen set.
This script copies images into the gold dataset layout, computes SHA-256, and
validates the emitted JSONL against the gold manifest schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REQUIRED_COLUMNS = {
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
    "annotator_id",
    "labeled_at",
    "labelers",
    "agreed",
    "tiebreak",
    "consent_scope",
    "notes",
    "storage_ref",
}

OOD_CONDITION = {
    "healthy_skin",
    "non_skin_pet",
    "human_skin",
    "other_species",
    "poor_quality",
    "unknown_ood",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-csv", type=Path, required=True, help="Adjudicated intake CSV.")
    parser.add_argument("--gold-root", type=Path, required=True, help="Output gold dataset root.")
    parser.add_argument("--dataset-version", default="gold-v0")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("docs/ml/schema/gold-manifest.schema.json"),
        help="Gold manifest row JSON Schema.",
    )
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    parser.add_argument(
        "--allow-existing-identical",
        action="store_true",
        help="Reuse existing destination files only when their SHA-256 matches the source.",
    )
    return parser.parse_args()


def parse_bool(value: str, *, field: str, line_no: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise SystemExit(f"Invalid boolean at line {line_no} field {field}: {value!r}")


def split_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace("|", ";").split(";") if item.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(csv_path: Path, value: str) -> Path:
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = csv_path.parent / path
    return path


def image_subdir(row: dict[str, str]) -> Path:
    ood_class = row["ood_class"].strip()
    condition = row["condition"].strip()
    if ood_class == "in_scope":
        return Path("images") / "in_scope" / condition
    return Path("images") / "ood" / ood_class


def clean_optional(value: str) -> str | None:
    value = value.strip()
    return value if value else None


def manifest_row(
    row: dict[str, str],
    *,
    csv_path: Path,
    gold_root: Path,
    dataset_version: str,
    line_no: int,
) -> tuple[dict[str, Any], Path, Path]:
    vet_confirmed = parse_bool(row["vet_confirmed"], field="vet_confirmed", line_no=line_no)
    if not vet_confirmed:
        raise SystemExit(f"Refusing non-vet-confirmed gold row at line {line_no}")

    labelers = split_list(row["labelers"])
    if not labelers:
        raise SystemExit(f"Missing adjudication labelers at line {line_no}")

    src = source_path(csv_path, row["source_path"])
    if not src.is_file():
        raise SystemExit(f"Missing source image at line {line_no}: {src}")

    image_id = row["image_id"].strip()
    suffix = src.suffix.lower() or ".jpg"
    relative_image = image_subdir(row) / f"{image_id}{suffix}"
    dst = gold_root / relative_image
    digest = sha256(src)
    storage_ref = row["storage_ref"].strip() or f"private://{dataset_version}/{relative_image.as_posix()}"

    out = {
        "imageId": image_id,
        "datasetVersion": dataset_version,
        "imagePath": relative_image.as_posix(),
        "storageRef": storage_ref,
        "sha256": digest,
        "source": row["source"].strip(),
        "license": row["license"].strip(),
        "species": row["species"].strip(),
        "breed": clean_optional(row["breed"]),
        "bodyRegion": row["body_region"].strip(),
        "condition": row["condition"].strip(),
        "oodClass": row["ood_class"].strip(),
        "qualityFlags": split_list(row["quality_flags"]),
        "vetConfirmed": True,
        "annotatorId": row["annotator_id"].strip(),
        "labeledAt": row["labeled_at"].strip(),
        "adjudication": {
            "labelers": labelers,
            "agreed": parse_bool(row["agreed"], field="agreed", line_no=line_no),
            "tiebreak": clean_optional(row["tiebreak"]),
        },
        "consentScope": row["consent_scope"].strip(),
        "notes": row["notes"].strip(),
        "neverTrain": True,
    }
    return out, src, dst


def read_intake(
    intake_csv: Path,
    gold_root: Path,
    dataset_version: str,
    schema_path: Path,
) -> list[tuple[dict[str, Any], Path, Path]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    rows: list[tuple[dict[str, Any], Path, Path]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_destinations: set[Path] = set()

    with intake_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise SystemExit(f"Intake CSV is missing columns: {', '.join(missing)}")

        for line_no, csv_row in enumerate(reader, start=2):
            row, src, dst = manifest_row(
                {key: value or "" for key, value in csv_row.items()},
                csv_path=intake_csv,
                gold_root=gold_root,
                dataset_version=dataset_version,
                line_no=line_no,
            )
            errors = sorted(validator.iter_errors(row), key=str)
            if errors:
                first = errors[0]
                path = ".".join(str(part) for part in first.path) or "<row>"
                raise SystemExit(f"Schema error at line {line_no} path {path}: {first.message}")
            if row["oodClass"] in OOD_CONDITION and row["condition"] != "unknown":
                raise SystemExit(f"OOD row must use condition=unknown at line {line_no}")
            if row["imageId"] in seen_ids:
                raise SystemExit(f"Duplicate image_id at line {line_no}: {row['imageId']}")
            if row["sha256"] in seen_hashes:
                raise SystemExit(f"Duplicate image content at line {line_no}: {row['sha256']}")
            if dst in seen_destinations:
                raise SystemExit(f"Duplicate destination at line {line_no}: {dst}")
            seen_ids.add(row["imageId"])
            seen_hashes.add(row["sha256"])
            seen_destinations.add(dst)
            rows.append((row, src, dst))

    if not rows:
        raise SystemExit(f"Intake CSV has no rows: {intake_csv}")
    return rows


def write_gold_dataset(
    rows: list[tuple[dict[str, Any], Path, Path]],
    *,
    gold_root: Path,
    manifest_name: str,
    allow_existing_identical: bool,
) -> Path:
    gold_root.mkdir(parents=True, exist_ok=True)
    manifest_path = gold_root / manifest_name
    if manifest_path.exists():
        raise SystemExit(f"Manifest already exists: {manifest_path}")

    for row, src, dst in rows:
        if dst.exists():
            if not allow_existing_identical or sha256(dst) != row["sha256"]:
                raise SystemExit(f"Destination already exists: {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row, _, _ in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    args = parse_args()
    rows = read_intake(args.intake_csv, args.gold_root, args.dataset_version, args.schema)
    manifest_path = write_gold_dataset(
        rows,
        gold_root=args.gold_root,
        manifest_name=args.manifest_name,
        allow_existing_identical=args.allow_existing_identical,
    )
    print(f"wrote {manifest_path}")
    print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
