# Gold Evaluation Set — spec

Owner: ML Lead · Task: `ll2` · Status: spec (collection not started)

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

```json
{
  "imageId": "gold_000123",
  "storageRef": "s3://.../gold/000123.jpg",
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
  "adjudication": { "labelers": ["a11", "a04"], "agreed": true, "tiebreak": null }
}
```

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

## Acceptance criteria

- ≥ 20 vet-confirmed images per condition (v0 unblock); ≥ 40 for v1.
- Every gold image has a complete provenance manifest row with `vetConfirmed: true`.
- OOD and quality buckets populated.
- Manifest hashed, versioned on HF, and excluded from training by hash.
- Enables the offline eval harness (`oej`) to report per-class recall, confusion
  matrix, ECE, and OOD AUROC without touching UI strings.
