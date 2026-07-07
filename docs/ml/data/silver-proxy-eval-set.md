# Silver Proxy Evaluation Set

Owner: ML Lead · Task: `petcare-triage-service-dpe` · Status: executable proxy flow

`silver-v0` is a **proxy evaluation set** for rehearsing the ML process when
real vet-confirmed images are unavailable. It can validate ingestion, leakage
guards, HF publication, offline evaluation commands, and relative model ranking.
It is not a clinical gold set and must not be used for accuracy claims.

## Boundary

- `gold-v0`: real vet-confirmed evidence only; `vetConfirmed=true`.
- `silver-v0`: simulated or non-vet adjudication; `vetConfirmed=false`,
  `validationTier=proxy`, and `adjudicationMode` records how labels were made.

Silver rows are still `neverTrain=true` because they act as an eval proxy. Do
not mix them into training data.

## Intake

Start from `silver-v0.intake.template.csv` and replace the template rows. Keep
raw candidate images outside git, for example under `ml/silver/intake/raw/`.
Example manifest rows live in `silver-v0.manifest.example.jsonl`; they are
schema examples only.

For the current Roboflow bootstrap mirror, the `test` split was excluded from
the CV train/val folds and can be used as a proxy rehearsal set. It is still
noisy uploader-labeled data, not vet-confirmed evidence, and only covers
`atopic_dermatitis`, `bacterial_pyoderma`, `fungal_malassezia`, and
`healthy_skin`.

Generate a deterministic silver intake CSV from that held-out split:

```bash
uv run python ml/create_silver_intake_from_manifest.py \
  --source-manifest ml/prepared/roboflow_dog_skin_disease_dataset_v2/manifest.jsonl \
  --out-csv ml/silver/intake/silver-v0.roboflow-test.csv \
  --split test \
  --max-per-bucket 20 \
  --labeler proxy_roboflow \
  --labeler reviewer_01
```

Build the proxy dataset:

```bash
uv run python ml/build_silver_manifest.py \
  --intake-csv ml/silver/intake/silver-v0.roboflow-test.csv \
  --silver-root ml/silver/silver-v0 \
  --dataset-version silver-v0
```

The builder refuses `vet_confirmed=true`, duplicate image IDs, duplicate image
hashes, missing source files, schema violations, or pre-existing destination
files. It writes `manifest.jsonl`, copies images into `images/...`, computes
SHA-256, and emits `private://silver-v0/...` storage refs when blank.

Audit coverage:

```bash
uv run python ml/audit_silver_intake.py \
  --intake-csv ml/silver/intake/silver-v0.roboflow-test.csv \
  --target-per-bucket 20
```

Run leakage checks before publishing:

```bash
uv run --with pillow --with imagehash python ml/check_gold_leakage.py \
  --gold-manifest ml/silver/silver-v0/manifest.jsonl \
  --gold-root ml/silver/silver-v0 \
  --training-manifest ml/prepared/roboflow_dog_skin_disease_dataset_v2_dedup_phash4/manifest.jsonl \
  --training-manifest ml/prepared/mendeley_5dbht54kw7_v1/manifest.jsonl \
  --phash-threshold 4 \
  --out ml/silver/silver-v0/leakage-report.json
```

The leakage checker is name-compatible with gold but works with any manifest
that has `imagePath`/`sha256`.

The first local rehearsal is documented in
[`../runs/silver-v0-roboflow-test-20260707.md`](../runs/silver-v0-roboflow-test-20260707.md).
It produced a usable process rehearsal but found pHash leakage against training
data, so it is not an independent evaluation set.

## Reporting Rule

Report silver metrics as:

> Proxy silver-v0 model-ranking result, not vet-confirmed clinical accuracy.

Any public or product-facing accuracy statement must wait for `gold-v0`.
