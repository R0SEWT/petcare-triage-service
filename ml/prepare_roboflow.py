#!/usr/bin/env python3
"""Normalize the Roboflow dog skin disease classification export.

Input is the Roboflow "folder" classification layout:

    <source>/train/<class>/*.jpg
    <source>/valid/<class>/*.jpg
    <source>/test/<class>/*.jpg

Output is a local PetCare manifest plus symlinked image layout. Raw and prepared
outputs are gitignored; the manifest is for local provenance and training only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}

CLASS_MAP = {
    "bacterial dermatosis": {
        "condition": "bacterial_pyoderma",
        "oodClass": "in_scope",
        "supervised": True,
    },
    "fungal infection": {
        "condition": "fungal_malassezia",
        "oodClass": "in_scope",
        "supervised": True,
    },
    "hypersensitivity dermatitis": {
        "condition": "atopic_dermatitis",
        "oodClass": "in_scope",
        "supervised": True,
    },
    "healthy": {
        "condition": "unknown",
        "oodClass": "healthy_skin",
        "supervised": False,
    },
    "Unlabeled": {
        "condition": "unknown",
        "oodClass": "unlabeled_source",
        "supervised": False,
    },
}

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("ml/_raw/roboflow_probe_folder"),
        help="Roboflow folder-format export.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ml/prepared/roboflow_dog_skin_disease_dataset_v2"),
        help="Prepared output directory. Must not already contain files unless --overwrite-links.",
    )
    parser.add_argument(
        "--overwrite-links",
        action="store_true",
        help="Replace existing symlinks in the prepared image layout.",
    )
    return parser.parse_args()


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

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            source_class = class_dir.name
            if source_class not in CLASS_MAP:
                raise SystemExit(f"Unsupported Roboflow class: {source_class}")
            mapping = CLASS_MAP[source_class]
            bucket = str(mapping["condition"])
            if mapping["oodClass"] == "healthy_skin":
                bucket = "healthy_skin"
            elif mapping["oodClass"] == "unlabeled_source":
                bucket = "unlabeled_source"

            for src in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
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
                        "dataset_id": "roboflow-dog-skin-disease-dataset",
                        "dataset_version": 2,
                        "source_url": "https://universe.roboflow.com/dog-skin-disease-dermatosis/dog-skin-disease-dataset",
                        "export_format": "folder",
                        "license": "CC BY 4.0 (Roboflow uploader-declared)",
                        "split": split,
                        "source_split": source_split,
                        "source_class": source_class,
                        "condition": mapping["condition"],
                        "oodClass": mapping["oodClass"],
                        "supervised_condition_example": mapping["supervised"],
                        "species": "dog",
                        "image_path": str(dest.relative_to(out_dir)),
                        "raw_path": str(rel_raw),
                        "sha256": sha256(src),
                        "vetConfirmed": False,
                        "notes": "Noisy Roboflow bootstrap data; not valid for gold evaluation.",
                    }
                )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (str(row["split"]), str(row["source_class"]), str(row["oodClass"]))
        counts[key] = counts.get(key, 0) + 1

    print(f"wrote {manifest_path}")
    print(f"rows {len(rows)}")
    for (split, source_class, ood_class), count in sorted(counts.items()):
        print(f"{split:5s} {source_class:30s} {ood_class:16s} {count:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
