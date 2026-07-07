import csv
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ml.build_gold_manifest import read_intake, write_gold_dataset

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
        "image_id": "gold_000001",
        "source": "vet_clinic_partner_A",
        "license": "consented",
        "species": "dog",
        "breed": "unknown",
        "body_region": "belly",
        "condition": "atopic_dermatitis",
        "ood_class": "in_scope",
        "quality_flags": "",
        "vet_confirmed": "true",
        "annotator_id": "vet_01",
        "labeled_at": "2026-07-07",
        "labelers": "vet_01;reviewer_01",
        "agreed": "true",
        "tiebreak": "",
        "consent_scope": "eval_only",
        "notes": "unit test row",
        "storage_ref": "",
    }


def test_build_gold_manifest_copies_image_and_validates_schema(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    schema_path = repo / "docs/ml/schema/gold-manifest.schema.json"
    source = tmp_path / "source.jpg"
    source.write_bytes(b"fake image bytes")
    intake = tmp_path / "intake.csv"
    write_intake(intake, valid_row("source.jpg"))

    rows = read_intake(intake, tmp_path / "gold-v0", "gold-v0", schema_path)
    manifest_path = write_gold_dataset(
        rows,
        gold_root=tmp_path / "gold-v0",
        manifest_name="manifest.jsonl",
        allow_existing_identical=False,
    )

    manifest_row = json.loads(manifest_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    validator.validate(manifest_row)
    assert manifest_row["neverTrain"] is True
    assert manifest_row["vetConfirmed"] is True
    assert (tmp_path / "gold-v0" / manifest_row["imagePath"]).read_bytes() == b"fake image bytes"


def test_build_gold_manifest_rejects_non_vet_confirmed_rows(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "source.jpg"
    source.write_bytes(b"fake image bytes")
    row = valid_row("source.jpg")
    row["vet_confirmed"] = "false"
    intake = tmp_path / "intake.csv"
    write_intake(intake, row)

    with pytest.raises(SystemExit, match="non-vet-confirmed"):
        read_intake(
            intake,
            tmp_path / "gold-v0",
            "gold-v0",
            repo / "docs/ml/schema/gold-manifest.schema.json",
        )
