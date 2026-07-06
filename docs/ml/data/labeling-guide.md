# Labeling Guide — PetCare dermatology triage

Owner: ML Lead · Task: `ll2` · Applies to: gold eval set first, training data later.

This guide makes labels reproducible and auditable. Label space and flags match
the contract (`../schema/labels.json`, `triage-response.schema.json`) so a
labeled image maps 1:1 to a persisted triage record.

## Who labels

Condition labels for the **gold eval set MUST be vet-confirmed** — assigned or
adjudicated by a licensed veterinarian (or vet student under supervision). Lay
labelers may pre-sort and flag image quality, but not finalize a condition.
Training-set labels may be lay-labeled with vet spot-checks; the gold set may not.

## Label fields (per image)

| Field | Values | Source of truth |
| --- | --- | --- |
| `condition` | `atopic_dermatitis`, `dermatophytosis`, `allergic_contact_dermatitis`, `fungal_malassezia`, `bacterial_pyoderma`, `unknown` | `../schema/labels.json` |
| `species` | `dog`, `cat` | contract input enum |
| `bodyRegion` | `ear`, `belly`, `paw`, `tail_base`, `flank`, `unknown` | contract input enum |
| `qualityFlags` | subset of `blur`, `too_dark`, `too_bright`, `obstruction`, `low_resolution` | matches `imageQuality.issues` |
| `oodClass` | `in_scope`, `healthy_skin`, `non_skin`, `human_skin`, `other_species` | for the OOD gate |
| `vetConfirmed` | `true` / `false` | required `true` for gold set |

## Condition cues (orientation, NOT clinical authority)

Brief visual orientation for labelers; the vet's judgment overrides. Each entry
notes the top confusion to force a deliberate differential.

- **atopic_dermatitis** — diffuse redness/itch pattern, often paws/face/belly;
  chronic, symmetric. Confuse with: allergic_contact_dermatitis.
- **allergic_contact_dermatitis** — reaction localized to contact area (belly,
  chin, paws); sharply demarcated. Confuse with: atopic_dermatitis.
- **dermatophytosis** (ringworm, **zoonotic**) — circular alopecia with scaling,
  often expanding ring. Confuse with: bacterial_pyoderma. When unsure, do not
  under-label — flag for vet review.
- **fungal_malassezia** — greasy, erythematous, often malodorous; ear canals,
  skin folds, tail base. Confuse with: bacterial_pyoderma.
- **bacterial_pyoderma** — pustules, papules, epidermal collarettes, crusting;
  can be serious. Confuse with: fungal_malassezia, dermatophytosis.
- **unknown** — use when no condition is confidently assignable, or image is
  out-of-scope. Never guess to fill a class.

> These are triage cues for annotation consistency, not diagnostic criteria. The
> product itself never presents a diagnosis (contract safety block).

## Adjudication & agreement

- **Two independent labelers** per image; disagreements go to a **vet tiebreak**.
- Target **Cohen's κ ≥ 0.6** condition agreement before scaling collection; if
  below, tighten cues and re-train labelers.
- Record `annotatorId` and `labeledAt` per label for auditability.

## Inclusion / exclusion

- **Include:** dog/cat, one primary lesion area, real photo, owner consent on file.
- **Exclude / route to OOD:** human skin, other species, non-skin photos, heavy
  filters, or images failing all quality checks.
- An excluded image is not discarded — it becomes an OOD/quality negative with
  the appropriate `oodClass` / `qualityFlags`.

## Quality flag rubric

- `blur` — lesion edges not resolvable at 100%.
- `too_dark` / `too_bright` — lesion color/texture unreadable.
- `obstruction` — hair, hand, or object covers >⅓ of the lesion.
- `low_resolution` — lesion region < ~128 px on the short side.
