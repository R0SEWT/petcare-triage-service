#!/usr/bin/env python3
"""Build a Roboflow bootstrap split filtered against reference manifests.

The source Roboflow export is useful as noisy bootstrap training data, but it
contains resized/augmented derivatives from the same image lineage as Mendeley.
This script removes exact SHA matches and perceptual-hash near matches against
reference manifests, then writes a new train/val-only manifest plus symlinked
image layout. The original raw/prepared exports are left untouched.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import imagehash
    from PIL import Image
except ImportError as exc:  # pragma: no cover - exercised by CLI users.
    raise SystemExit(
        "Missing dependency. Install ML requirements or run with "
        "`uv run --with pillow --with imagehash python ml/filter_roboflow_dedup.py`."
    ) from exc


DEFAULT_CANDIDATE = Path("ml/prepared/roboflow_dog_skin_disease_dataset_v2/manifest.jsonl")
DEFAULT_REFERENCE = Path("ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl")
DEFAULT_OUT = Path("ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4")
TRAINING_SPLITS = {"train", "val"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        help="Root for candidate image_path values. Defaults to candidate manifest parent.",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        action="append",
        default=[DEFAULT_REFERENCE],
        help="Reference manifest to exclude exact/perceptual matches against. Repeatable.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=4,
        help="Remove candidate rows whose nearest reference pHash distance is <= this value.",
    )
    parser.add_argument(
        "--keep-split",
        action="append",
        choices=sorted(TRAINING_SPLITS | {"test"}),
        help="Candidate split to keep. Defaults to train and val only.",
    )
    parser.add_argument(
        "--overwrite-links",
        action="store_true",
        help="Replace existing symlinks under the output image layout.",
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


def image_path(root: Path, row: dict[str, Any]) -> Path:
    image_value = row.get("image_path") or row.get("imagePath")
    if not image_value:
        raise SystemExit(f"Missing image path for manifest row {row.get('_manifest')}:{row.get('_line')}")
    path = root / str(image_value)
    if not path.is_file():
        raise SystemExit(f"Missing image for manifest row {row.get('_manifest')}:{row.get('_line')}: {path}")
    return path


def phash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))


def build_reference_index(reference_manifests: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_sha: dict[str, dict[str, Any]] = {}
    refs: list[dict[str, Any]] = []
    for manifest in reference_manifests:
        root = manifest.parent
        for row in load_jsonl(manifest):
            path = image_path(root, row)
            digest = str(row.get("sha256") or "")
            record = {
                "manifest": str(manifest),
                "line": row["_line"],
                "image_path": row.get("image_path") or row.get("imagePath"),
                "sha256": digest,
                "condition": row.get("condition"),
                "oodClass": row.get("oodClass"),
                "dataset_id": row.get("dataset_id"),
                "dataset_version": row.get("dataset_version"),
                "source_id": row.get("source_folder") or row.get("source_filename") or row.get("raw_path"),
                "phash": phash(path),
            }
            refs.append(record)
            if digest and digest not in by_sha:
                by_sha[digest] = record
    return by_sha, refs


def nearest_phash(candidate_hash: imagehash.ImageHash, refs: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    nearest_distance = 10**9
    nearest_ref = refs[0]
    for ref in refs:
        distance = int(candidate_hash - ref["phash"])
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_ref = ref
    return nearest_distance, nearest_ref


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def match_record(match_type: str, ref: dict[str, Any], distance: int | None = None) -> dict[str, Any]:
    record = {
        "match_type": match_type,
        "reference_manifest": ref["manifest"],
        "reference_line": ref["line"],
        "reference_image_path": ref["image_path"],
        "reference_sha256": ref["sha256"],
        "reference_condition": ref["condition"],
        "reference_oodClass": ref["oodClass"],
        "reference_dataset_id": ref["dataset_id"],
        "reference_dataset_version": ref["dataset_version"],
        "reference_source_id": ref["source_id"],
    }
    if distance is not None:
        record["phash_distance"] = distance
        record["reference_phash"] = str(ref["phash"])
    return record


def write_symlink(src: Path, dest: Path, overwrite_links: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if not overwrite_links:
            raise SystemExit(f"{dest} exists; pass --overwrite-links to refresh symlinks")
        if not dest.is_symlink():
            raise SystemExit(f"{dest} exists and is not a symlink; refusing to replace it")
        dest.unlink()
    dest.symlink_to(src.resolve())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists; choose a new --out directory")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    keep_splits = set(args.keep_split or sorted(TRAINING_SPLITS))
    candidate_root = args.candidate_root or args.candidate_manifest.parent
    out_dir = args.out
    manifest_out = out_dir / "manifest.jsonl"
    removed_out = out_dir / "removed.jsonl"
    report_out = out_dir / "filter_report.json"
    if manifest_out.exists() or removed_out.exists() or report_out.exists():
        raise SystemExit(f"{out_dir} already has filter outputs; choose a new --out directory")

    candidate_rows = load_jsonl(args.candidate_manifest)
    reference_by_sha, references = build_reference_index(args.reference_manifest)
    if not references:
        raise SystemExit("No reference rows loaded")

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    kept_counts: Counter[str] = Counter()
    removed_counts: Counter[str] = Counter()

    for row in candidate_rows:
        candidate_split = str(row["split"])
        src_path = image_path(candidate_root, row)
        base_row = clean_row(row)

        if candidate_split not in keep_splits:
            removed_row = {
                **base_row,
                "filter_status": "removed",
                "filter_reason": "excluded_split",
                "filter_detail": f"split={candidate_split} not in keep_splits={sorted(keep_splits)}",
            }
            removed.append(removed_row)
            removed_counts[f"excluded_split:{candidate_split}:{row['source_class']}"] += 1
            continue

        exact_ref = reference_by_sha.get(str(row.get("sha256") or ""))
        candidate_hash = phash(src_path)
        nearest_distance, nearest_ref = nearest_phash(candidate_hash, references)

        if exact_ref is not None:
            removed_row = {
                **base_row,
                "filter_status": "removed",
                "filter_reason": "exact_sha_reference_match",
                "candidate_phash": str(candidate_hash),
                "nearest_reference_phash_distance": nearest_distance,
                "duplicate_match": match_record("exact_sha", exact_ref),
            }
            removed.append(removed_row)
            removed_counts[f"exact_sha_reference_match:{row['source_class']}"] += 1
            continue

        if nearest_distance <= args.phash_threshold:
            removed_row = {
                **base_row,
                "filter_status": "removed",
                "filter_reason": "phash_reference_near_duplicate",
                "candidate_phash": str(candidate_hash),
                "nearest_reference_phash_distance": nearest_distance,
                "duplicate_match": match_record("phash", nearest_ref, nearest_distance),
            }
            removed.append(removed_row)
            removed_counts[f"phash_reference_near_duplicate:{row['source_class']}"] += 1
            continue

        dest = out_dir / str(row["image_path"])
        write_symlink(src_path, dest, args.overwrite_links)
        kept_row = {
            **base_row,
            "filter_status": "kept",
            "filter_id": "roboflow_v2_mendeley_phash4_trainval",
            "dedup_reference_manifests": [str(path) for path in args.reference_manifest],
            "dedup_phash_threshold": args.phash_threshold,
            "nearest_reference_phash_distance": nearest_distance,
            "candidate_phash": str(candidate_hash),
            "notes": (
                str(row.get("notes") or "")
                + " Filtered for exact SHA and pHash near-duplicates against reference manifests."
            ).strip(),
        }
        kept.append(kept_row)
        kept_counts[f"{row['split']}:{row['source_class']}"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_out, kept)
    write_jsonl(removed_out, removed)
    report = {
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_root": str(candidate_root),
        "reference_manifests": [str(path) for path in args.reference_manifest],
        "out_dir": str(out_dir),
        "keep_splits": sorted(keep_splits),
        "phash_threshold": args.phash_threshold,
        "candidate_rows": len(candidate_rows),
        "reference_rows": len(references),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "kept_counts": dict(sorted(kept_counts.items())),
        "removed_counts": dict(sorted(removed_counts.items())),
    }
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {manifest_out}")
    print(f"wrote {removed_out}")
    print(f"wrote {report_out}")
    print(f"kept_rows {len(kept)}")
    print(f"removed_rows {len(removed)}")
    for key, count in sorted(kept_counts.items()):
        print(f"kept {key} {count}")
    for key, count in sorted(removed_counts.items()):
        print(f"removed {key} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
