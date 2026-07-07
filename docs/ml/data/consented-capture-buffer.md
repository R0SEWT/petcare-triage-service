# Consented capture buffer and HF sync

Status: service-side v0
Task: `petcare-triage-service-d9b`

## Decision

When `POST /api/triage/analyze` completes successfully and
`consentForModelImprovement=true`, the service writes the uploaded image plus a
JSONL metadata row to a local capture buffer. Capture is best-effort: failures
are logged and never change the triage response.

The buffer is raw training material, not evaluation data. Rows start with
nullable feedback/correction fields until a return-flow exists:

- `feedback.helpful`
- `feedback.ownerCorrection`
- `feedback.vetConfirmedCondition`
- `feedback.outcomeNotes`

## Local Layout

Default buffer:

```text
data/bronze/triage-captures/
  README.md
  metadata.jsonl
  images/YYYY/MM/DD/cap_*.jpg
```

Override with:

```bash
PETCARE_CAPTURE_BUFFER_DIR=/path/to/captures
PETCARE_CAPTURE_ENABLED=false  # optional kill switch
```

The default path is gitignored. Do not commit raw captures.

## Privacy Guardrails

- Raw `petId` and `clientRequestId` are not stored; SHA-256 hashes are stored.
- `symptomNotes` free text is not stored; only presence and length are stored.
- Uploaded images may still contain owner-identifying content, so sync only to a
  private Hugging Face dataset unless public release is explicitly approved.
- Captures are not vet-confirmed and must not be used for clinical accuracy
  claims without a separate review/adjudication process.

## HF Sync

Dry-run:

```bash
uv run python ml/sync_capture_buffer_to_hf.py \
  --repo-id <hf-user-or-org>/petcare-triage-captures-private \
  --dry-run
```

Create the private dataset repo if needed and upload:

```bash
uv run python ml/sync_capture_buffer_to_hf.py \
  --repo-id <hf-user-or-org>/petcare-triage-captures-private \
  --create-repo
```

The script uses the `hf` CLI authentication already configured in the
environment, for example `HF_TOKEN` or `hf auth login`. It does not print tokens.
