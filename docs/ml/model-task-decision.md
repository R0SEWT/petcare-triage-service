# Decision Record: YOLOv8 in classification mode

Status: Revised v0.2 (2026-07-06) · Owner: ML Lead
Supersedes v0.1 ("generic classifier"). Reason for change: keep the YOLOv8
branding already present throughout PetCare-Web, without paying for detection.

## Decision

**Use YOLOv8 in classification mode (`yolov8-cls`).** The model consumes one
pet-skin photo and returns a calibrated probability over the canonical condition
set (`schema/labels.json`) plus top-k. It does **not** return bounding boxes.

The key point: **YOLOv8 is not only an object detector.** Ultralytics ships
`yolov8n/s/m-cls` classification variants. Choosing the classification variant
lets us satisfy three things at once:

- **Branding stays truthful.** The frontend already says "Analizando con
  YOLOv8…" (`dashboard.new-analysis.tsx`) and `mock-data.ts` names YOLOv8. With
  `yolov8-cls` that copy is accurate — no frontend churn.
- **Contract is unchanged.** `prediction` = condition + confidence + top-k, no
  bbox field. Nothing in `schema/` moves.
- **Labeling stays cheap.** Image-level labels only — critical given there is no
  labeled pet-derm dataset yet (see `data/dataset-scouting.md`).

Model name going forward: `petcare-derm-yolov8-cls`.

## Classification vs. detection — why cls now

| Criterion | `yolov8-cls` (chosen) | YOLOv8 detection |
| --- | --- | --- |
| Matches contract output (no bbox) | ✅ exact | ❌ needs a new `regions` field |
| Labeling cost | ✅ image-level | ❌ per-lesion boxes, ~3–5× effort, expert annotators |
| Branding ("YOLOv8" in UI) | ✅ truthful | ✅ truthful |
| Data availability (none labeled yet) | ✅ fastest to a baseline | ❌ box labels don't exist |
| Calibration (confidence drives urgency) | ✅ standard (temp scaling, ECE) | ⚠️ box-score calibration murkier |

The product question is *which condition + how urgent*, not *where on the body*
(body region is already a user-supplied input). Detection solves a problem we
don't have and taxes the scarcest resources — labeled data and annotation time.

## Consequences

- Baseline = transfer-learned `yolov8-cls` on a curated set (Task `oej`).
- Contract keeps `prediction.topK`, no `boundingBoxes`. No schema change.
- OOD / lesion-presence handled by the OOD + image-quality gates, not a detector.
- Mock `model.name` = `petcare-derm-yolov8-cls` (`services/triage-mock/app.py`).

## Revisit → detection only if

1. Product needs to **show users where** the lesion is (annotated overlay), **and**
2. We can fund **box-level labeling** by someone qualified, **and**
3. Localization measurably improves triage trust or accuracy in testing.

Then detection is **additive** (a second head in the same YOLOv8 family) behind a
contract minor bump that adds an optional `regions` field — not a rewrite.
