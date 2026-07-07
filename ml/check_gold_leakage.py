#!/usr/bin/env python3
"""Check frozen gold images for leakage into training manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        action="append",
        required=True,
        help="Training/prepared manifest to compare against. Repeatable.",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=None,
        help="Also flag perceptual-hash matches at or below this distance. Requires pillow + imagehash.",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON report path.")
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


def row_image_path(root: Path, row: dict[str, Any]) -> Path:
    image_value = row.get("imagePath") or row.get("image_path")
    if not image_value:
        raise SystemExit(f"Missing image path in {row.get('_manifest')}:{row.get('_line')}")
    return root / str(image_value)


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("imageId") or row.get("image_id") or row.get("raw_path") or row.get("image_path"))


def load_training_rows(training_manifests: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in training_manifests:
        for row in load_jsonl(manifest):
            row["_root"] = manifest.parent
            rows.append(row)
    return rows


def exact_matches(gold_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    training_by_sha: dict[str, list[dict[str, Any]]] = {}
    for row in training_rows:
        digest = str(row.get("sha256") or "")
        if digest:
            training_by_sha.setdefault(digest, []).append(row)

    matches: list[dict[str, Any]] = []
    for gold in gold_rows:
        digest = str(gold.get("sha256") or "")
        if not digest:
            continue
        for train in training_by_sha.get(digest, []):
            matches.append(
                {
                    "matchType": "exact_sha",
                    "sha256": digest,
                    "goldImageId": row_id(gold),
                    "goldManifest": gold["_manifest"],
                    "goldLine": gold["_line"],
                    "goldImagePath": gold.get("imagePath"),
                    "trainingManifest": train["_manifest"],
                    "trainingLine": train["_line"],
                    "trainingImagePath": train.get("image_path") or train.get("imagePath"),
                    "trainingCondition": train.get("condition"),
                    "trainingSplit": train.get("split"),
                }
            )
    return matches


def load_phash_dependencies() -> tuple[Any, Any]:
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised by CLI users.
        raise SystemExit(
            "Missing dependency for pHash leakage checks. Run with "
            "`uv run --with pillow --with imagehash python ml/check_gold_leakage.py ...` "
            "or omit --phash-threshold for exact-SHA checks only."
        ) from exc
    return imagehash, Image


def image_phash(path: Path, imagehash: Any, image_module: Any) -> Any:
    if not path.is_file():
        raise SystemExit(f"Missing image for pHash check: {path}")
    with image_module.open(path) as image:
        return imagehash.phash(image.convert("RGB"))


def phash_matches(
    gold_rows: list[dict[str, Any]],
    gold_root: Path,
    training_rows: list[dict[str, Any]],
    threshold: int,
) -> list[dict[str, Any]]:
    imagehash, image_module = load_phash_dependencies()
    training_hashes: list[tuple[dict[str, Any], Any]] = []
    for row in training_rows:
        training_hashes.append((row, image_phash(row_image_path(row["_root"], row), imagehash, image_module)))

    matches: list[dict[str, Any]] = []
    for gold in gold_rows:
        gold_hash = image_phash(row_image_path(gold_root, gold), imagehash, image_module)
        for train, train_hash in training_hashes:
            distance = int(gold_hash - train_hash)
            if distance <= threshold:
                matches.append(
                    {
                        "matchType": "phash",
                        "phashDistance": distance,
                        "goldImageId": row_id(gold),
                        "goldManifest": gold["_manifest"],
                        "goldLine": gold["_line"],
                        "goldImagePath": gold.get("imagePath"),
                        "goldPhash": str(gold_hash),
                        "trainingManifest": train["_manifest"],
                        "trainingLine": train["_line"],
                        "trainingImagePath": train.get("image_path") or train.get("imagePath"),
                        "trainingCondition": train.get("condition"),
                        "trainingSplit": train.get("split"),
                        "trainingPhash": str(train_hash),
                    }
                )
    return matches


def leakage_report(
    gold_manifest: Path,
    gold_root: Path,
    training_manifests: list[Path],
    phash_threshold: int | None,
) -> dict[str, Any]:
    gold_rows = load_jsonl(gold_manifest)
    training_rows = load_training_rows(training_manifests)
    exact = exact_matches(gold_rows, training_rows)
    phash = (
        phash_matches(gold_rows, gold_root, training_rows, phash_threshold)
        if phash_threshold is not None
        else []
    )
    return {
        "goldManifest": str(gold_manifest),
        "goldRows": len(gold_rows),
        "trainingManifests": [str(path) for path in training_manifests],
        "trainingRows": len(training_rows),
        "phashThreshold": phash_threshold,
        "leakageFound": bool(exact or phash),
        "exactMatches": exact,
        "phashMatches": phash,
    }


def main() -> int:
    args = parse_args()
    report = leakage_report(args.gold_manifest, args.gold_root, args.training_manifest, args.phash_threshold)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 1 if report["leakageFound"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
