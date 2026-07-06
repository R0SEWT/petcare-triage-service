# petcare-triage-service - AI Agent Instructions

## Domain / Scientific Context

- **Problem**: PetCare needs dermatology triage for dog skin images without putting clinical logic in the frontend.
- **Outcome / target**: A contract-conformant ML/backend service that can return safe triage states, abstain when uncertain, and support future model improvement from curated data.
- **Data provenance**: Current baseline uses public Kaggle data mapped to coarse canonical labels. Credible evaluation requires a separate vet-verified gold set. Future training data should include independent licensed sources plus consented app captures, with provenance and leakage controls.

## Architecture

```bash
# mock/service path
uv sync --extra dev
uv run pytest -q
cd services/triage-mock && uv run uvicorn app:app --reload --port 8000

# ML path (GPU box for training)
cd ml
python prepare_data.py --source-dir <downloaded>/Dogs --out ./prepared
python train_yolo_cls.py --data ./prepared --device 0 --epochs 50
```

Data flow:

```text
PetCare-Web upload -> triage API contract -> mock or model-backed inference
                                      -> optional consented capture buffer
                                      -> batch sync to private HF dataset
                                      -> review/promotion for training
```

## Key Files

| File | Purpose |
| --- | --- |
| `docs/ml/inference-contract.md` | Human-readable contract and API behavior. |
| `docs/ml/schema/` | OpenAPI, JSON Schemas, labels, urgency policy. Source of truth for service/frontend integration. |
| `services/triage-mock/app.py` | FastAPI mock implementation of `POST /api/triage/analyze`. |
| `services/triage-mock/smoke_test.py` | Contract smoke test for all mock scenarios. |
| `ml/label_map.json` | Coarse v0 dataset-to-canonical label mapping. |
| `ml/prepare_data.py` | Stdlib dataset preparation for Ultralytics classification layout. |
| `ml/train_yolo_cls.py` | YOLOv8-cls training entrypoint for GPU environments. |
| `docs/ml/data/` | Dataset scouting, gold eval set, and labeling guide. |

## Data Conventions

- Raw downloads, prepared datasets, runs, and weights are gitignored.
- Dataset manifests should include source URL/ref, license, acquisition date, transform, label mapping, split, and dedup notes.
- Keep the gold eval set frozen and separate from training/capture data.
- Do not train on gold eval data.
- App captures start as `raw` and require review before promotion.

## Development Conventions

- Use `bd` for all task tracking. Run `bd prime` at session start.
- Use `uv` for repo-level dev/runtime dependencies.
- Keep GPU-only dependencies in `ml/requirements.txt` unless CI or the API needs them.
- Run `uv run ruff check .` and `uv run pytest -q` after code changes.
- Use the existing contract schemas instead of duplicating label or urgency logic.
- Treat ML metrics as evidence only when provenance, split, and evaluation set are recorded.



<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
