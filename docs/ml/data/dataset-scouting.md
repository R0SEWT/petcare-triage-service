# Dataset Scouting — HF Hub survey (2026-07-06)

Owner: ML Lead · Task: `ll2` · Scope: 5 canonical conditions + `unknown`, dog/cat.

## Headline finding

**There is no off-the-shelf labeled canine/feline dermatology-condition image
dataset on the Hugging Face Hub, and no existing pet-derm model.** Every image
dataset returned for "skin disease" / "dermatology" is **human** dermatology;
every "veterinary" dataset is text/QA/tabular. This confirms the plan's risk #2
(data scarcity): **the gold eval set must be built, not downloaded.**

## What the Hub *does* offer (and how we reuse it)

| Use | Candidates | License | Notes |
| --- | --- | --- | --- |
| **OOD negatives / "is-this-pet-skin" gate** | `microsoft/cats_vs_dogs`, `Voxel51/Stanford-Dogs-Imbalanced`, `Bingsu/Cat_and_Dog`, `sasha/dog-food` | mixed (cc0 / unknown) | Healthy whole-animal + hard non-skin negatives. Strong for the OOD gate, useless as condition labels. |
| **Transfer-learning pretrain (human derm domain)** | `HawkFranklin-Research/SCIN-Dermatology-Raw-Images` (6,517 imgs), `Digital-Dermatology/CleanPatrick` (Fitzpatrick17k-based) | MIT / **CC-BY-NC-4.0** | Human skin. Domain-adjacent texture/lesion features for backbone warm-up only. CleanPatrick is **non-commercial** — fine for portfolio/learning, not for a commercial product. |
| **Human OOD examples** | same as above | — | Human skin is a useful "confidently-not-a-pet" OOD probe. |

Links:
- https://hf.co/datasets/HawkFranklin-Research/SCIN-Dermatology-Raw-Images
- https://hf.co/datasets/Digital-Dermatology/CleanPatrick
- https://hf.co/datasets/microsoft/cats_vs_dogs
- https://hf.co/datasets/Voxel51/Stanford-Dogs-Imbalanced

## Verdict

1. **Condition labels come from a purpose-built collection**, not the Hub. Primary
   source: a vet-clinic partnership (real, consented photos with vet-confirmed
   labels). Secondary: licensed teaching atlases / curated web images, each
   vet-confirmed before entering the gold set. See `gold-eval-set.md`.
2. **Reuse the Hub for the OOD gate and pretraining only.** Pull healthy pet
   images as OOD negatives and (optionally) warm-start the backbone on human
   derm texture. Never let human/healthy images carry one of the 5 condition
   labels.
3. **This scarcity is the project's moat.** A small, vet-verified pet-derm eval
   set is a genuinely scarce asset and the portfolio centerpiece — lean into it.

## Off-Hub candidates (real canine-derm image data) — found 2026-07-06

Unlike HF, Roboflow and Kaggle **do** have canine skin-disease image datasets,
YOLO-ready. These unblock a **fast first baseline** (`yolov8-cls`) while the
vet-verified gold set is being built.

| Source | What | Use |
| --- | --- | --- |
| [Roboflow: dog-skin-disease-dataset](https://universe.roboflow.com/dog-skin-disease-dermatosis/dog-skin-disease-dataset) | Classification: healthy, fungal, bacterial dermatosis, hypersensitivity dermatitis | strongest cls candidate |
| [Roboflow: litespy/dog-skin-diseases](https://universe.roboflow.com/litespy-l22hu/dog-skin-diseases) | Detection, 618 imgs + pre-trained model/API | reference; detection (not our path) |
| [Kaggle: youssefmohmmed/dogs-skin-diseases-image-dataset](https://www.kaggle.com/datasets/youssefmohmmed/dogs-skin-diseases-image-dataset) | Image dataset | training candidate |
| [Kaggle: yashmotiani/dogs-skin-disease-dataset](https://www.kaggle.com/datasets/yashmotiani/dogs-skin-disease-dataset) | Image dataset | training candidate |
| [Kaggle: smadive/pet-disease-images](https://www.kaggle.com/datasets/smadive/pet-disease-images) | Pet disease images | training candidate |
| Published study ([IJSDR2507201](https://ijsdr.org/papers/IJSDR2507201.pdf)) | 4,315 imgs, 6 classes, YOLOv8, 92.38% acc | prior art / method reference |

### ⚠️ Label-mapping gap (must handle before training)

Their taxonomies are **not** our 5 canonical labels. Typical classes:
Fungal Infections, Dermatitis, Demodicosis, Healthy, Hypersensitivity, Ringworm,
bacterial dermatosis. Mapping to `schema/labels.json`:

- `Ringworm` → `dermatophytosis`
- `bacterial dermatosis` → `bacterial_pyoderma`
- `Fungal Infections` → `fungal_malassezia` (approximate — verify)
- `Hypersensitivity` / `Dermatitis` → `atopic_dermatitis` / `allergic_contact_dermatitis` (ambiguous — needs judgment)
- `Healthy` → OOD/negative (`oodClass: healthy_skin`)
- `Demodicosis` → out of scope → `unknown` or drop

These are web-scraped, often noisy, and mislabeled. **Good enough for a first
baseline, not for credible evaluation.** Eval numbers only count on the
vet-verified gold set (`gold-eval-set.md`).

## Still worth doing

- Confirm license/usage terms on the chosen Roboflow/Kaggle set before committing.
- Perceptual-hash dedup between any off-the-shelf training data and the gold set.
- Vet-school outreach for the gold set remains the highest-value data action.

## 2026-07-07 update — independent training expansion

The first local `silver-v0` rehearsal from the Roboflow test split is useful for
pipeline validation but not independent evaluation: exact SHA leakage was clean,
while pHash threshold 4 found near-duplicates against train/val. See
`../runs/silver-v0-roboflow-test-20260707.md`.

For training expansion, prioritize independent/semi-independent sources that
cover labels missing from the current bootstrap, especially `ringworm` →
`dermatophytosis`. The current shortlist lives in
`independent-training-candidates.md` and
`independent-training-candidates.json`.
