import json
from pathlib import Path

from ml.check_gold_leakage import leakage_report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_check_gold_leakage_flags_exact_sha_matches(tmp_path: Path):
    gold_manifest = tmp_path / "gold" / "manifest.jsonl"
    training_manifest = tmp_path / "train" / "manifest.jsonl"
    digest = "a" * 64
    write_jsonl(
        gold_manifest,
        [{"imageId": "gold_000001", "imagePath": "images/gold_000001.jpg", "sha256": digest}],
    )
    write_jsonl(
        training_manifest,
        [{"image_path": "images/train_000001.jpg", "sha256": digest, "condition": "atopic_dermatitis"}],
    )

    report = leakage_report(gold_manifest, tmp_path / "gold", [training_manifest], phash_threshold=None)

    assert report["leakageFound"] is True
    assert report["matchSummary"]["totalMatches"] == 1
    assert report["matchSummary"]["uniqueGoldRowsMatched"] == 1
    assert report["matchSummary"]["byMatchType"] == {"exact_sha": 1}
    assert report["exactMatches"][0]["matchType"] == "exact_sha"
    assert report["exactMatches"][0]["goldImageId"] == "gold_000001"
    assert report["phashMatches"] == []


def test_check_gold_leakage_allows_distinct_sha_rows(tmp_path: Path):
    gold_manifest = tmp_path / "gold" / "manifest.jsonl"
    training_manifest = tmp_path / "train" / "manifest.jsonl"
    write_jsonl(
        gold_manifest,
        [{"imageId": "gold_000001", "imagePath": "images/gold_000001.jpg", "sha256": "a" * 64}],
    )
    write_jsonl(
        training_manifest,
        [{"image_path": "images/train_000001.jpg", "sha256": "b" * 64, "condition": "atopic_dermatitis"}],
    )

    report = leakage_report(gold_manifest, tmp_path / "gold", [training_manifest], phash_threshold=None)

    assert report["leakageFound"] is False
    assert report["matchSummary"]["totalMatches"] == 0
    assert report["matchSummary"]["uniqueGoldRowsMatched"] == 0
    assert report["exactMatches"] == []
