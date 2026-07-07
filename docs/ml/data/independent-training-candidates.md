# Independent Training Data Candidates

Owner: ML Lead · Task: `petcare-triage-service-yu3` · Status: shortlist

This shortlist is for **training/proxy expansion only**. None of these sources
are vet-confirmed PetCare gold data. Every imported image still needs provenance
rows, exact SHA/pHash dedup against existing training/silver/gold manifests, and
label mapping notes before it can influence model training.

Structured registry: `independent-training-candidates.json`.

## Current Gap

The local `silver-v0` rehearsal from Roboflow test split covers:

- `atopic_dermatitis`
- `bacterial_pyoderma`
- `fungal_malassezia`
- `healthy_skin`

It does not cover:

- `dermatophytosis`
- `allergic_contact_dermatitis`
- `non_skin_pet`
- `human_skin`
- `other_species`
- `poor_quality`

## Priority Acquisition Order

1. **Roboflow Kaivlya Dogs skin disease**
   - URL: https://universe.roboflow.com/kaivlya/dogs-skin-disease-fxh4x
   - Why: broadest classification taxonomy found so far: `ringworm`,
     `flea_allergy`, `hotspot`, `mange`, `demodicosis`, `Hypersensitivity`,
     `Bacterial_dermatosis`, `Fungal_infections`, and `Healthy`.
   - Result: not exportable right now. The public page reports `Dataset versions
     0` and `Models 0`, so there is no stable Universe export version.

2. **Roboflow dog-skin-disease-prediction v3**
   - URL: https://universe.roboflow.com/majorproject-kopqr/dog-skin-disease-prediction
   - Why: direct `ringworm` class for `dermatophytosis`; `flea_allergy` /
     `hotspot` are useful allergy/inflammation proxies.
   - Use: noisy training/proxy only.

3. **Roboflow Gian Ringworm/Demodicosis detection v1**
   - URL: https://universe.roboflow.com/gian-ocdnp/dog-skin-diseases-b7b5y-jvz7i
   - Why: larger reported set with `Ringworm`; `Demodicosis` is useful
     out-of-scope dermatology.
   - Use: crop/convert detection annotations into classification candidates.

4. **Roboflow dog-skin-disease-detection v1**
   - URL: https://universe.roboflow.com/myprojects-zsnac/dog-skin-disease-detection-6pgvk
   - Why: another small `ringworm` / allergy-like source for cross-source
     comparison after dedup.

5. **Kaggle smadive/pet-disease-images**
   - URL: https://www.kaggle.com/datasets/smadive/pet-disease-images
   - Why: reported CC0; possible pet OOD / poor-quality examples.
   - Use: inspect folder taxonomy first; do not map to condition labels blindly.

6. **HF OOD sources**
   - Human skin: `HawkFranklin-Research/SCIN-Dermatology-Raw-Images`
   - Non-skin pets: `microsoft/cats_vs_dogs`
   - Use only for OOD/pretraining. Human and whole-pet images must never carry
     one of the five pet dermatology condition labels.

## Import Gate

Before any candidate becomes training data:

1. Create raw and normalized manifests with source URL, license, source class,
   target mapping, `vetConfirmed=false`, and `supervised_condition_example`.
2. Run exact SHA and pHash dedup against:
   - `ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4/manifest.jsonl`
   - `ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl`
   - `ml/silver/silver-v0/manifest.jsonl` when present
   - future `ml/gold/gold-v0/manifest.jsonl`
3. Keep ambiguous labels as proxy/unknown instead of forcing canonical labels.
4. Publish only private HF datasets until license/provenance is stronger.

## Decision

Kaivlya is not currently exportable through a stable version. Continue with
`roboflow-dog-skin-disease-prediction-v3`, which is the smallest useful way to
add `ringworm`/`dermatophytosis` signal and test cross-source dedup. See
`roboflow-dog-skin-disease-prediction-v3.md`.
