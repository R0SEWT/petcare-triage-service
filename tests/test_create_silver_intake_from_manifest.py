import csv
from pathlib import Path

from ml.create_silver_intake_from_manifest import FIELDNAMES, intake_rows


def test_create_silver_intake_samples_split_with_bucket_cap(tmp_path: Path):
    source_root = tmp_path / "prepared"
    for relative in [
        "images/test/atopic/a.jpg",
        "images/test/atopic/b.jpg",
        "images/test/healthy/h.jpg",
    ]:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    rows = [
        {
            "split": "test",
            "image_path": "images/test/atopic/a.jpg",
            "condition": "atopic_dermatitis",
            "oodClass": "in_scope",
            "species": "dog",
            "license": "CC BY 4.0",
        },
        {
            "split": "test",
            "image_path": "images/test/atopic/b.jpg",
            "condition": "atopic_dermatitis",
            "oodClass": "in_scope",
            "species": "dog",
            "license": "CC BY 4.0",
        },
        {
            "split": "test",
            "image_path": "images/test/healthy/h.jpg",
            "condition": "unknown",
            "oodClass": "healthy_skin",
            "species": "dog",
            "license": "CC BY 4.0",
        },
        {
            "split": "train",
            "image_path": "images/train/atopic/t.jpg",
            "condition": "atopic_dermatitis",
            "oodClass": "in_scope",
            "species": "dog",
            "license": "CC BY 4.0",
        },
    ]

    selected, report = intake_rows(
        rows,
        source_root=source_root,
        out_csv=tmp_path / "silver" / "intake.csv",
        split="test",
        max_per_bucket=1,
        id_prefix="silver_test",
        adjudication_mode="simulated",
        annotator_id="proxy_01",
        labelers=["proxy_01", "reviewer_01"],
        labeled_at="2026-07-07",
        consent_scope="eval_only",
    )

    assert len(selected) == 2
    assert report["countsByBucket"] == {"atopic_dermatitis": 1, "healthy_skin": 1}
    assert selected[0]["image_id"] == "silver_test_000001"
    assert selected[0]["vet_confirmed"] == "false"
    assert selected[0]["adjudication_mode"] == "simulated"
    assert selected[0]["labelers"] == "proxy_01;reviewer_01"
    assert set(selected[0]) == set(FIELDNAMES)


def test_create_silver_intake_rows_are_csv_serializable(tmp_path: Path):
    source_root = tmp_path / "prepared"
    image = source_root / "images/test/healthy/h.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")

    selected, _ = intake_rows(
        [
            {
                "split": "test",
                "image_path": "images/test/healthy/h.jpg",
                "condition": "unknown",
                "oodClass": "healthy_skin",
                "species": "dog",
                "license": "CC BY 4.0",
            }
        ],
        source_root=source_root,
        out_csv=tmp_path / "silver" / "intake.csv",
        split="test",
        max_per_bucket=20,
        id_prefix="silver_test",
        adjudication_mode="simulated",
        annotator_id="proxy_01",
        labelers=["proxy_01"],
        labeled_at="2026-07-07",
        consent_scope="eval_only",
    )

    with (tmp_path / "out.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(selected)
