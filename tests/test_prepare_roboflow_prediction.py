import json
from pathlib import Path

from ml.prepare_roboflow_prediction import main


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image")


def test_prepare_roboflow_prediction_maps_ringworm_and_ood(tmp_path: Path, monkeypatch):
    source = tmp_path / "raw"
    for split in ["train", "valid", "test"]:
        write_image(source / split / "ringworm" / f"{split}_ringworm.jpg")
        write_image(source / split / "mange" / f"{split}_mange.jpg")
    out = tmp_path / "prepared"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_roboflow_prediction.py",
            "--source-dir",
            str(source),
            "--out",
            str(out),
        ],
    )

    assert main() == 0
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    ringworm = [row for row in rows if row["source_class"] == "ringworm"]
    mange = [row for row in rows if row["source_class"] == "mange"]

    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert all(row["condition"] == "dermatophytosis" for row in ringworm)
    assert all(row["oodClass"] == "in_scope" for row in ringworm)
    assert all(row["supervised_condition_example"] is True for row in ringworm)
    assert all(row["condition"] == "unknown" for row in mange)
    assert all(row["oodClass"] == "unknown_ood" for row in mange)
    assert all(row["supervised_condition_example"] is False for row in mange)
    assert all((out / row["image_path"]).is_symlink() for row in rows)
