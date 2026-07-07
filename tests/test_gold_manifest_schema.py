import json
from pathlib import Path

from jsonschema import Draft202012Validator


def test_gold_manifest_example_matches_schema():
    repo = Path(__file__).resolve().parents[1]
    schema = json.loads((repo / "docs/ml/schema/gold-manifest.schema.json").read_text())
    validator = Draft202012Validator(schema)
    example = repo / "docs/ml/data/gold-v0.manifest.example.jsonl"

    rows = [json.loads(line) for line in example.read_text().splitlines() if line.strip()]
    assert rows
    for row in rows:
        validator.validate(row)


def test_silver_manifest_example_matches_schema():
    repo = Path(__file__).resolve().parents[1]
    schema = json.loads((repo / "docs/ml/schema/silver-manifest.schema.json").read_text())
    validator = Draft202012Validator(schema)
    example = repo / "docs/ml/data/silver-v0.manifest.example.jsonl"

    rows = [json.loads(line) for line in example.read_text().splitlines() if line.strip()]
    assert rows
    for row in rows:
        validator.validate(row)
