from ml.audit_gold_intake import audit_rows


def test_audit_gold_intake_reports_bucket_deficits():
    rows = [
        {
            "condition": "atopic_dermatitis",
            "oodClass": "in_scope",
            "species": "dog",
            "qualityFlags": [],
        },
        {
            "condition": "unknown",
            "oodClass": "healthy_skin",
            "species": "cat",
            "qualityFlags": ["blur"],
        },
    ]

    summary = audit_rows(rows, target_per_bucket=1)

    assert summary["complete"] is False
    assert summary["counts"]["conditions"] == {"atopic_dermatitis": 1}
    assert summary["counts"]["ood"] == {"healthy_skin": 1}
    assert summary["counts"]["species"] == {"cat": 1, "dog": 1}
    assert summary["counts"]["qualityFlags"] == {"blur": 1}
    assert summary["deficits"]["conditions"]["atopic_dermatitis"] == 0
    assert summary["deficits"]["conditions"]["dermatophytosis"] == 1
    assert summary["deficits"]["ood"]["healthy_skin"] == 0
    assert summary["deficits"]["ood"]["poor_quality"] == 1


def test_audit_gold_intake_marks_complete_when_all_targets_met():
    rows = []
    for condition in [
        "atopic_dermatitis",
        "dermatophytosis",
        "allergic_contact_dermatitis",
        "fungal_malassezia",
        "bacterial_pyoderma",
    ]:
        rows.append(
            {
                "condition": condition,
                "oodClass": "in_scope",
                "species": "dog",
                "qualityFlags": [],
            }
        )
    for ood_class in ["healthy_skin", "non_skin_pet", "human_skin", "other_species", "poor_quality"]:
        rows.append(
            {
                "condition": "unknown",
                "oodClass": ood_class,
                "species": "unknown",
                "qualityFlags": [],
            }
        )

    summary = audit_rows(rows, target_per_bucket=1)

    assert summary["complete"] is True
