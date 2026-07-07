from collections import Counter

import pytest

from ml.audit_mendeley_labels import missing_distribution


def test_missing_distribution_subtracts_label_file_counts_from_reported_totals():
    expected = {
        "Bacterial_dermatosis": 23,
        "Fungal_infections": 19,
        "Hypersensitivity_allergic_dermatosis": 23,
        "Healthy": 30,
    }
    observed = Counter(
        {
            "Bacterial_dermatosis": 12,
            "Fungal_infections": 11,
            "Hypersensitivity_allergic_dermatosis": 13,
            "Healthy": 26,
        }
    )

    assert missing_distribution(expected, observed) == {
        "Bacterial_dermatosis": 11,
        "Fungal_infections": 8,
        "Hypersensitivity_allergic_dermatosis": 10,
        "Healthy": 4,
    }


def test_missing_distribution_rejects_observed_counts_above_reported_totals():
    with pytest.raises(ValueError, match="above expected total"):
        missing_distribution({"Healthy": 30}, Counter({"Healthy": 31}))
