import json
from pathlib import Path

from ml.prepare_hf_vet_derm import CLASS_MAP, main


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image")


def test_class_map_keeps_ambiguous_dermatitis_out_of_supervised_conditions():
    assert CLASS_MAP["Dermatitis"]["condition"] == "unknown"
    assert CLASS_MAP["Dermatitis"]["supervised"] is False
    assert CLASS_MAP["Dermatitis"]["oodClass"] == "ambiguous_dermatitis_proxy"


def test_class_map_maps_ringworm_to_dermatophytosis_proxy():
    assert CLASS_MAP["ringworm"]["condition"] == "dermatophytosis"
    assert CLASS_MAP["ringworm"]["oodClass"] == "in_scope"
    assert CLASS_MAP["ringworm"]["supervised"] is True


def test_prepare_hf_vet_derm_writes_manifest_and_symlinks(tmp_path: Path, monkeypatch):
    source = tmp_path / "raw"
    rows = [
        {
            "image": "images/Dermatitis_0000.jpg",
            "prefix": "question",
            "suffix": "answer",
            "class": "Dermatitis",
            "split": "train",
        },
        {
            "image": "images/ringworm_0000.jpg",
            "prefix": "question",
            "suffix": "answer",
            "class": "ringworm",
            "split": "valid",
        },
    ]
    source.mkdir()
    (source / "paligemma_dataset.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    for row in rows:
        write_image(source / row["image"])
    out = tmp_path / "prepared"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_hf_vet_derm.py",
            "--source-dir",
            str(source),
            "--out",
            str(out),
        ],
    )

    assert main() == 0
    manifest_rows = [
        json.loads(line) for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    dermatitis = next(row for row in manifest_rows if row["source_class"] == "Dermatitis")
    ringworm = next(row for row in manifest_rows if row["source_class"] == "ringworm")

    assert dermatitis["condition"] == "unknown"
    assert dermatitis["supervised_condition_example"] is False
    assert ringworm["condition"] == "dermatophytosis"
    assert ringworm["split"] == "val"
    assert all((out / row["image_path"]).is_symlink() for row in manifest_rows)
