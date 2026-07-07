import csv
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ml.build_silver_manifest import read_intake, write_silver_dataset

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


def write_intake(path: Path, row: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def valid_row(source_path: str) -> dict[str, str]:
    return {
        "source_path": source_path,
        "image_id": "silver_000001",
        "source": "roboflow_proxy",
        "license": "source_license_here",
        "species": "dog",
        "breed": "unknown",
        "body_region": "belly",
        "condition": "atopic_dermatitis",
        "ood_class": "in_scope",
        "quality_flags": "",
        "vet_confirmed": "false",
        "adjudication_mode": "simulated",
        "annotator_id": "proxy_01",
        "labeled_at": "2026-07-07",
        "labelers": "proxy_01;reviewer_01",
        "agreed": "true",
        "tiebreak": "",
        "consent_scope": "eval_only",
        "notes": "unit test proxy row",
        "storage_ref": "",
    }


def test_build_silver_manifest_marks_proxy_and_not_vet_confirmed(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    schema_path = repo / "docs/ml/schema/silver-manifest.schema.json"
    source = tmp_path / "source.jpg"
    source.write_bytes(b"fake image bytes")
    intake = tmp_path / "intake.csv"
    write_intake(intake, valid_row("source.jpg"))

    rows = read_intake(intake, tmp_path / "silver-v0", "silver-v0", schema_path)
    manifest_path = write_silver_dataset(
        rows,
        silver_root=tmp_path / "silver-v0",
        manifest_name="manifest.jsonl",
        allow_existing_identical=False,
    )

    manifest_row = json.loads(manifest_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    validator.validate(manifest_row)
    assert manifest_row["validationTier"] == "proxy"
    assert manifest_row["vetConfirmed"] is False
    assert manifest_row["neverTrain"] is True
    assert (tmp_path / "silver-v0" / manifest_row["imagePath"]).read_bytes() == b"fake image bytes"


def test_build_silver_manifest_rejects_vet_confirmed_rows(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "source.jpg"
    source.write_bytes(b"fake image bytes")
    row = valid_row("source.jpg")
    row["vet_confirmed"] = "true"
    intake = tmp_path / "intake.csv"
    write_intake(intake, row)

    with pytest.raises(SystemExit, match="vet_confirmed=false"):
        read_intake(
            intake,
            tmp_path / "silver-v0",
            "silver-v0",
            repo / "docs/ml/schema/silver-manifest.schema.json",
        )
