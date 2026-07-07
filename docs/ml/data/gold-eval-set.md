# Gold Evaluation Set — spec

Owner: ML Lead · Task: `petcare-triage-service-gy0` · Status: executable spec (collection not started)

The gold set is a **small, frozen, vet-verified** test set that measures the
classifier + OOD gate. It is never trained on. It is the single asset that lets
any accuracy claim be credible (see `dataset-scouting.md` for why it must be
built, not downloaded).

## Composition (target)

| Bucket | Target | Notes |
| --- | ---: | --- |
| Each of the 5 conditions | ≥ 40–60 vet-confirmed images | Balanced across dog/cat where possible; span body regions. |
| In-scope subtotal | ~250–300 | 5 × ~50. |
| OOD negatives | ~100 | healthy pet skin, non-skin pet photos, human skin, other species (from Hub — see scouting). |
| Quality-fail examples | ~40 | blur/dark/obstruction, to test the quality gate. |
| **Total** | **~400–450** | Small on purpose; quality over volume. |

Start smaller (≥20/class) to unblock the first eval, then grow to target. A
tiny vet-verified set beats a large noisy one.

## Provenance manifest (per image)

Stored as JSONL alongside images; fields mirror the contract's persistence
requirements so eval records and production records share a shape.
Rows must validate against
[`../schema/gold-manifest.schema.json`](../schema/gold-manifest.schema.json).

```json
{
  "imageId": "gold_000123",
  "datasetVersion": "gold-v0",
  "imagePath": "images/in_scope/allergic_contact_dermatitis/gold_000123.jpg",
  "storageRef": "private://gold-v0/images/in_scope/allergic_contact_dermatitis/gold_000123.jpg",
  "sha256": "<64 lowercase hex chars>",
  "source": "vet_clinic_partner_A | licensed_atlas | curated_web | hf:<repo>",
  "license": "consented | CC-BY-NC-4.0 | MIT | ...",
  "species": "dog",
  "breed": "golden_retriever",
  "bodyRegion": "belly",
  "condition": "allergic_contact_dermatitis",
  "oodClass": "in_scope",
  "qualityFlags": [],
  "vetConfirmed": true,
  "annotatorId": "vet_07",
  "labeledAt": "2026-07-06",
  "adjudication": { "labelers": ["a11", "a04"], "agreed": true, "tiebreak": null },
  "consentScope": "eval_only",
  "neverTrain": true
}
```

Example rows live in `gold-v0.manifest.example.jsonl`. They are schema examples
only and are not part of the frozen set.

## Intake workflow

Use `gold-v0.intake.template.csv` for adjudicated candidate rows. Replace the
template rows before use; they are not valid collection data. Keep the raw
candidate images outside git, for example under `ml/gold/intake/raw/`.

The intake CSV must be explicit about provenance and review status before any
image becomes part of the frozen set:

- `source_path`: source image path, relative to the CSV or absolute.
- `image_id`: stable `gold_...` identifier.
- `source`, `license`, `consent_scope`: provenance and permission.
- `species`, `breed`, `body_region`, `condition`, `ood_class`, `quality_flags`:
  label fields matching `gold-manifest.schema.json`.
- `vet_confirmed`, `annotator_id`, `labeled_at`, `labelers`, `agreed`,
  `tiebreak`: adjudication trail.
- `notes`, `storage_ref`: optional context and external storage pointer.

Build the private gold dataset directory from an adjudicated CSV:

```bash
uv run python ml/build_gold_manifest.py \
  --intake-csv ml/gold/intake/gold-v0.adjudicated.csv \
  --gold-root ml/gold/gold-v0 \
  --dataset-version gold-v0
```

The builder refuses rows without `vet_confirmed=true`, missing source files,
duplicate image IDs, duplicate image hashes, schema violations, or pre-existing
destination files. It writes `ml/gold/gold-v0/manifest.jsonl`, copies images
into `images/...`, computes SHA-256, sets `neverTrain: true`, and emits
`private://gold-v0/...` storage refs when the CSV leaves `storage_ref` blank.

Audit progress before freezing the manifest:

```bash
uv run python ml/audit_gold_intake.py \
  --intake-csv ml/gold/intake/gold-v0.adjudicated.csv \
  --target-per-bucket 20
```

The audit reports per-condition/OOD/species/quality counts and target deficits.
Use `--fail-on-deficit` in automation once the CSV is expected to be complete.

## Sourcing sequence

1. **Vet-clinic partnership** (primary) — real, consented photos with vet-confirmed
   labels. Best quality and the best portfolio story.
2. **Licensed teaching atlases** — with permission; vet-confirm before inclusion.
3. **Curated web images** — carefully licensed, each vet-confirmed. Lowest trust;
   use to fill thin classes only.
4. **HF Hub** — OOD negatives and (optional) pretraining only; never for the 5
   condition labels.

## Discipline

- **Frozen & never trained on.** Hash the manifest; any change bumps the eval-set
  version. Training pipelines must exclude these `imageId`s by hash.
- **Versioned on HF datasets** as a private/gated dataset; `datasetVersion` in the
  contract references it.
- **Leakage guard:** no image (or near-duplicate) may appear in both gold and
  training. Run perceptual-hash dedup across the boundary.
- **Executable gate:** `ml/evaluate_gold.py` refuses rows without
  `vetConfirmed: true`, `neverTrain: true`, present image files, and matching
  SHA-256 hashes.

## Offline eval harness

Run only after the frozen image directory and manifest exist:

```bash
uv run --extra dev python -m pytest tests/test_gold_manifest_schema.py
uv run python ml/evaluate_gold.py \
  --manifest ml/gold/gold-v0/manifest.jsonl \
  --image-root ml/gold/gold-v0 \
  --checkpoint ml/runs/petcare-derm-yolov8-cls/cv5-yolov8n-fold_0/weights/best.pt \
  --checkpoint ml/runs/petcare-derm-yolov8-cls/cv5-yolov8n-fold_1/weights/best.pt \
  --checkpoint ml/runs/petcare-derm-yolov8-cls/cv5-yolov8n-fold_2/weights/best.pt \
  --checkpoint ml/runs/petcare-derm-yolov8-cls/cv5-yolov8n-fold_3/weights/best.pt \
  --checkpoint ml/runs/petcare-derm-yolov8-cls/cv5-yolov8n-fold_4/weights/best.pt
```

Current bootstrap checkpoints do not implement a real OOD/image-quality gate.
The harness therefore reports in-scope condition accuracy and healthy-skin
negative accuracy separately, while counting non-skin/human/other-species and
poor-quality rows as present but not scored for condition accuracy. For scored
rows it emits confusion, per-label precision/recall/F1, abstention coverage at
the configured confidence threshold, and approximate ECE.

## Acceptance criteria

- ≥ 20 vet-confirmed images per condition (v0 unblock); ≥ 40 for v1.
- Every gold image has a complete provenance manifest row with `vetConfirmed: true`.
- OOD and quality buckets populated.
- Manifest hashed, versioned on HF, and excluded from training by hash.
- Enables the offline eval harness (`oej`) to report per-class recall, confusion
  matrix, ECE, and OOD AUROC without touching UI strings.
