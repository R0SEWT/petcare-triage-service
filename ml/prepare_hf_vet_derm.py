#!/usr/bin/env python3
"""Normalize shrayyyy/vet-derm-dataset into a PetCare training-candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET_ID = "hf-shrayyyy-vet-derm-dataset"
DATASET_REPO = "shrayyyy/vet-derm-dataset"
DATASET_REVISION = "69ed4ac2f4ee5abbc4eca18d7e5d9f01835d63d9"
SOURCE_URL = "https://huggingface.co/datasets/shrayyyy/vet-derm-dataset"
SOURCE_LINEAGE = "Kaggle youssefmohmmed/dogs-skin-diseases-image-dataset"
LICENSE = "Apache-2.0 (HF dataset card; inherited from stated Kaggle source)"
SPLIT_MAP = {"train": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}

CLASS_MAP = {
    "Dermatitis": {
        "condition": "unknown",
        "oodClass": "ambiguous_dermatitis_proxy",
        "supervised": False,
        "label_quality": "ambiguous_proxy",
        "notes": "Dermatitis is too broad to map to atopic vs allergic/contact without review.",
    },
    "Fungal_infections": {
        "condition": "fungal_malassezia",
        "oodClass": "in_scope",
        "supervised": True,
        "label_quality": "noisy_proxy",
        "notes": "Fungal_infections is a coarse noisy proxy for fungal_malassezia.",
    },
    "Healthy": {
        "condition": "unknown",
        "oodClass": "healthy_skin",
        "supervised": False,
        "label_quality": "negative_proxy",
        "notes": "Healthy is useful as negative/OOD skin, not as a condition example.",
    },
    "Hypersensitivity": {
        "condition": "atopic_dermatitis",
        "oodClass": "in_scope",
        "supervised": True,
        "label_quality": "weak_proxy",
        "notes": "Hypersensitivity is a weak allergy/atopy proxy; not vet-confirmed PetCare gold.",
    },
    "demodicosis": {
        "condition": "unknown",
        "oodClass": "unknown_ood",
        "supervised": False,
        "label_quality": "out_of_scope",
        "notes": "Demodicosis is out of current PetCare condition scope.",
    },
    "ringworm": {
        "condition": "dermatophytosis",
        "oodClass": "in_scope",
        "supervised": True,
        "label_quality": "noisy_proxy",
        "notes": "Ringworm maps to dermatophytosis as noisy training signal.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("ml/_raw/hf_vet_derm_dataset"),
        help="Directory containing README.md, paligemma_dataset.jsonl, and images/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ml/prepared/hf_vet_derm_dataset"),
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


def load_rows(jsonl_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"image", "class", "split"} - set(row)
            if missing:
                raise SystemExit(f"Missing {sorted(missing)} at {jsonl_path}:{line_no}")
            rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir
    out_dir = args.out
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        raise SystemExit(f"{manifest_path} already exists; choose a new --out")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    seen_images: set[str] = set()
    for index, source_row in enumerate(load_rows(source_dir / "paligemma_dataset.jsonl"), start=1):
        source_class = source_row["class"]
        if source_class not in CLASS_MAP:
            raise SystemExit(f"Unsupported class at row {index}: {source_class}")
        source_split = source_row["split"]
        if source_split not in SPLIT_MAP:
            raise SystemExit(f"Unsupported split at row {index}: {source_split}")
        raw_rel = Path(source_row["image"])
        if raw_rel.as_posix() in seen_images:
            raise SystemExit(f"Duplicate source image in JSONL: {raw_rel}")
        seen_images.add(raw_rel.as_posix())
        src = source_dir / raw_rel
        if not src.is_file():
            raise SystemExit(f"Missing source image at row {index}: {src}")

        mapping = CLASS_MAP[source_class]
        split = SPLIT_MAP[source_split]
        bucket = bucket_for(mapping)
        dest = out_dir / "images" / split / bucket / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not args.overwrite_links:
                raise SystemExit(f"{dest} exists; pass --overwrite-links to refresh")
            if not dest.is_symlink():
                raise SystemExit(f"{dest} exists and is not a symlink; refusing to replace it")
            dest.unlink()
        dest.symlink_to(src.resolve())
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
                "source_url": SOURCE_URL,
                "source_lineage": SOURCE_LINEAGE,
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
                "raw_path": raw_rel.as_posix(),
                "sha256": sha256(src),
                "vetConfirmed": False,
                "vlm_prefix": source_row.get("prefix", ""),
                "vlm_suffix": source_row.get("suffix", ""),
                "notes": mapping["notes"],
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    counts: dict[tuple[str, str, str, str, str], int] = {}
    for row in rows:
        key = (
            str(row["split"]),
            str(row["source_class"]),
            str(row["condition"]),
            str(row["oodClass"]),
            str(row["label_quality"]),
        )
        counts[key] = counts.get(key, 0) + 1

    print(f"wrote {manifest_path}")
    print(f"rows {len(rows)}")
    for (split, source_class, condition, ood_class, label_quality), count in sorted(counts.items()):
        print(
            f"{split:5s} {source_class:20s} {condition:24s} "
            f"{ood_class:28s} {label_quality:16s} {count:5d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
