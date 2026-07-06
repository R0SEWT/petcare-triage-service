#!/usr/bin/env python3
"""Run a small Roboflow Universe smoke test against local probe images.

This script intentionally does not download datasets. It sends selected local
probe images to Roboflow model endpoints when ROBOFLOW_API_KEY is available and
writes prediction JSONL into ml/runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_MODELS = [
    {
        "name": "taxonomy_aligned_classifier",
        "task": "classify",
        "model_id": "dog-skin-disease-dataset/2",
    },
    {
        "name": "clinical_named_classifier",
        "task": "classify",
        "model_id": "dog-skin-disease-prediction/3",
    },
    {
        "name": "lesion_detector",
        "task": "detect",
        "model_id": "dog-skin-diseases/1",
    },
]


DEFAULT_BUCKETS = [
    "bacterial_pyoderma",
    "fungal_malassezia",
    "atopic_dermatitis",
    "healthy_skin",
    "unlabeled_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl"),
        help="Prepared Mendeley manifest JSONL.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("ml/prepared/mendeley_5dbht54kw7_v1"),
        help="Root containing manifest image paths.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ml/runs/roboflow_smoke"),
        help="Ignored output directory for JSONL predictions.",
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=2,
        help="Number of Mendeley probes per bucket.",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        choices=DEFAULT_BUCKETS,
        help="Bucket to sample. Defaults to all smoke-test buckets.",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="TASK:MODEL_ID",
        help="Override model list, e.g. classify:dog-skin-disease-dataset/2.",
    )
    parser.add_argument(
        "--extra-image",
        action="append",
        metavar="LABEL:PATH",
        help="Add an extra probe image, e.g. non_skin:tmp/dog.jpg.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and print probes without calling Roboflow.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds.")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bucket_for(row: dict[str, Any]) -> str:
    if row.get("oodClass") == "healthy_skin":
        return "healthy_skin"
    if row.get("oodClass") == "unlabeled_source":
        return "unlabeled_source"
    return str(row.get("condition") or "unknown")


def select_probes(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    buckets: list[str],
    per_bucket: int,
    extra_images: list[str] | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = bucket_for(row)
        if bucket in buckets:
            grouped[bucket].append(row)

    probes: list[dict[str, Any]] = []
    for bucket in buckets:
        for row in grouped.get(bucket, [])[:per_bucket]:
            image_path = dataset_root / row["image_path"]
            probes.append(
                {
                    "probe_id": f"mendeley:{bucket}:{row['source_folder']}:{row['source_filename']}",
                    "bucket": bucket,
                    "expected_condition": row.get("condition"),
                    "expected_oodClass": row.get("oodClass"),
                    "source_label": row.get("source_label"),
                    "sha256": row.get("sha256"),
                    "image_path": str(image_path),
                }
            )

    for item in extra_images or []:
        if ":" not in item:
            raise ValueError(f"--extra-image must be LABEL:PATH, got {item!r}")
        label, raw_path = item.split(":", 1)
        probes.append(
            {
                "probe_id": f"extra:{label}:{Path(raw_path).name}",
                "bucket": label,
                "expected_condition": None,
                "expected_oodClass": label,
                "source_label": None,
                "sha256": None,
                "image_path": raw_path,
            }
        )

    missing = [probe["image_path"] for probe in probes if not Path(probe["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError("Missing probe images:\n" + "\n".join(missing))
    return probes


def parse_models(raw_models: list[str] | None) -> list[dict[str, str]]:
    if not raw_models:
        return DEFAULT_MODELS
    models: list[dict[str, str]] = []
    for raw in raw_models:
        if ":" not in raw:
            raise ValueError(f"--model must be TASK:MODEL_ID, got {raw!r}")
        task, model_id = raw.split(":", 1)
        if task not in {"classify", "detect"}:
            raise ValueError(f"Unsupported model task {task!r}; expected classify or detect")
        models.append({"name": model_id.replace("/", "_"), "task": task, "model_id": model_id})
    return models


def endpoint_for(task: str, model_id: str, api_key: str) -> str:
    host = {"classify": "classify.roboflow.com", "detect": "detect.roboflow.com"}[task]
    return f"https://{host}/{model_id}?{urlencode({'api_key': api_key})}"


def call_roboflow(model: dict[str, str], image_path: Path, api_key: str, timeout: int) -> dict[str, Any]:
    with image_path.open("rb") as handle:
        payload = handle.read()

    request = Request(
        endpoint_for(model["task"], model["model_id"], api_key),
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    started = time.time()
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "latency_ms": round((time.time() - started) * 1000),
            "error": body,
        }
    except URLError as exc:
        return {
            "ok": False,
            "status": None,
            "latency_ms": round((time.time() - started) * 1000),
            "error": str(exc.reason),
        }

    try:
        parsed: Any = json.loads(body)
    except json.JSONDecodeError:
        parsed = {"raw": body}
    return {
        "ok": 200 <= status < 300,
        "status": status,
        "latency_ms": round((time.time() - started) * 1000),
        "response": parsed,
    }


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest)
    buckets = args.bucket or DEFAULT_BUCKETS
    probes = select_probes(rows, args.dataset_root, buckets, args.per_bucket, args.extra_image)
    models = parse_models(args.model)

    print(f"selected_probes={len(probes)}")
    for probe in probes:
        print(
            "probe",
            probe["probe_id"],
            probe["expected_condition"],
            probe["expected_oodClass"],
            probe["image_path"],
        )

    if args.dry_run:
        return 0

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set; run with --dry-run or export the key.", file=sys.stderr)
        return 2

    run_dir = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / "predictions.jsonl"
    metadata_path = run_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(
            {
                "models": models,
                "probe_count": len(probes),
                "manifest": str(args.manifest),
                "dataset_root": str(args.dataset_root),
                "api_key_present": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for probe in probes:
            for model in models:
                result = call_roboflow(model, Path(probe["image_path"]), api_key, args.timeout)
                record = {"probe": probe, "model": model, "result": result}
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    "result",
                    probe["probe_id"],
                    model["model_id"],
                    "ok" if result["ok"] else "fail",
                    result.get("status"),
                    f"{result['latency_ms']}ms",
                )

    print(f"wrote={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
