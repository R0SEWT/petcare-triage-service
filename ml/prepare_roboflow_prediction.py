#!/usr/bin/env python3
"""Normalize the Roboflow dog-skin-disease-prediction v3 folder export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}

DATASET_ID = "roboflow-dog-skin-disease-prediction"
DATASET_VERSION = 3
SOURCE_URL = "https://universe.roboflow.com/majorproject-kopqr/dog-skin-disease-prediction"
LICENSE = "CC BY 4.0 (Roboflow uploader-declared)"

CLASS_MAP = {
    "ringworm": {
        "condition": "dermatophytosis",
        "oodClass": "in_scope",
        "supervised": True,
        "label_quality": "noisy_proxy",
        "notes": "Roboflow ringworm mapped to dermatophytosis as noisy training/proxy signal.",
    },
    "flea_allergy": {
        "condition": "allergic_contact_dermatitis",
        "oodClass": "in_scope",
        "supervised": True,
        "label_quality": "weak_proxy",
        "notes": "Flea allergy is only a weak allergy/contact-dermatitis proxy; do not use for eval claims.",
    },
    "hotspot": {
        "condition": "unknown",
        "oodClass": "unknown_ood",
        "supervised": False,
        "label_quality": "out_of_scope",
        "notes": "Hotspot/acute moist dermatitis is not a canonical PetCare class; keep as OOD/proxy only.",
    },
    "mange": {
        "condition": "unknown",
        "oodClass": "unknown_ood",
        "supervised": False,
        "label_quality": "out_of_scope",
        "notes": "Mange is out of current PetCare scope; useful as dermatology OOD only.",
    },
}

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("ml/_raw/roboflow_dog_skin_disease_prediction_v3"),
        help="Roboflow folder-format export.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ml/prepared/roboflow_dog_skin_disease_prediction_v3"),
    )
    parser.add_argument("--overwrite-links", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bucket_for(mapping: dict[str, object]) -> str:
    if mapping["oodClass"] == "in_scope":
        return str(mapping["condition"])
    return str(mapping["oodClass"])


def normalize_class_name(name: str) -> str:
    return name.strip().replace(" ", "_").lower()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir
    out_dir = args.out
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not args.overwrite_links:
        raise SystemExit(f"{manifest_path} already exists; pass --overwrite-links to refresh")

    rows: list[dict[str, object]] = []
    for source_split, split in SPLIT_MAP.items():
        split_dir = source_dir / source_split
        if not split_dir.is_dir():
            raise SystemExit(f"Missing split directory: {split_dir}")

        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            source_class = class_dir.name
            class_key = normalize_class_name(source_class)
            if class_key not in CLASS_MAP:
                raise SystemExit(f"Unsupported Roboflow class: {source_class}")
            mapping = CLASS_MAP[class_key]
            bucket = bucket_for(mapping)

            for src in sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMG_EXTS):
                rel_raw = src.relative_to(source_dir)
                dest = out_dir / "images" / split / bucket / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() or dest.is_symlink():
                    if not args.overwrite_links:
                        raise SystemExit(f"{dest} exists; pass --overwrite-links to refresh")
                    dest.unlink()
                dest.symlink_to(src.resolve())
                rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "dataset_version": DATASET_VERSION,
                        "source_url": SOURCE_URL,
                        "export_format": "folder",
                        "license": LICENSE,
                        "split": split,
                        "source_split": source_split,
                        "source_class": source_class,
                        "condition": mapping["condition"],
                        "oodClass": mapping["oodClass"],
                        "supervised_condition_example": mapping["supervised"],
                        "label_quality": mapping["label_quality"],
                        "species": "dog",
                        "image_path": str(dest.relative_to(out_dir)),
                        "raw_path": str(rel_raw),
                        "sha256": sha256(src),
                        "vetConfirmed": False,
                        "notes": mapping["notes"],
                    }
                )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        key = (
            str(row["split"]),
            str(row["source_class"]),
            str(row["condition"]),
            str(row["oodClass"]),
        )
        counts[key] = counts.get(key, 0) + 1

    print(f"wrote {manifest_path}")
    print(f"rows {len(rows)}")
    for (split, source_class, condition, ood_class), count in sorted(counts.items()):
        print(f"{split:5s} {source_class:20s} {condition:28s} {ood_class:12s} {count:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
