# petcare-triage-service Agent Instructions

## Safety

- Prefer read-only analysis before making changes.
- Do not use sudo unless explicitly requested.
- Do not delete files, model weights, datasets, or generated artifacts without explicit approval.
- Do not read secrets: SSH private keys, tokens, cookies, wallets, browser profiles, credential stores, `.env` files, Kaggle credentials, Hugging Face tokens, or cloud credentials.
- When auditing the machine, inspect metadata and paths, not secret contents.

## Workflow

- First produce a short plan.
- Separate confirmed evidence from inference.
- Use `bd` (beads) for task tracking. Run `bd prime` when starting a new session.
- Create or claim a bead before writing code.
- Prefer small, reversible patches.
- Before modifying files, explain the intended diff.
- After changes, run the smallest relevant verification command.

## Repository Ownership

This repo owns the PetCare ML/backend surface:

- `docs/ml/`: inference contract, executable schemas, label taxonomy, model/data decisions.
- `services/triage-mock/`: FastAPI contract-conformant mock service.
- `ml/`: YOLOv8-cls training/data-prep scripts and model provenance.
- Future work here: dataset acquisition, capture buffer/sync, HF dataset publishing, OOD gates, abstention/calibration, evaluation harnesses.

`PetCare-Web` owns UI, upload UX, consent UI, and API consumption. Do not move frontend concerns into this repo.

## Development Commands

- `uv sync --extra dev`: install service/dev dependencies from `pyproject.toml`.
- `uv run ruff check .`: lint Python.
- `uv run pytest -q`: run repository tests, including the mock smoke test.
- `cd services/triage-mock && uv run uvicorn app:app --reload --port 8000`: run the mock API locally.
- `cd services/triage-mock && uv run python smoke_test.py`: run the contract smoke test directly.
- `cd ml && python prepare_data.py --source-dir <downloaded>/Dogs --out ./prepared`: verify dataset mapping without GPU.
- `cd ml && python train_yolo_cls.py --data ./prepared --device 0 --epochs 50`: train only on a GPU box with ML deps installed.

## Python / ML Conventions

- Prefer `uv` for repo-level Python dependencies.
- Keep heavyweight training dependencies in `ml/requirements.txt` unless they are needed by CI or the service runtime.
- Keep raw data, prepared data, model weights, and run outputs out of git.
- Treat reported metrics as directional unless evaluated on the vet-verified frozen gold set.
- Preserve leakage discipline: training/capture datasets must not be mixed into the frozen gold evaluation set.
- Prefer structured manifests (`jsonl`, `yaml`, `json`) over ad hoc notes for dataset provenance.

## Hugging Face / Dataset Rules

- Use private HF datasets/repos for captures and intermediate model artifacts unless the user explicitly approves public release.
- Record source, license, transform, dedup status, and label mapping for every imported dataset.
- Do not upload PII or owner-identifying image content intentionally. For the demo stage, keep consent/disclaimer minimal; implement real PII handling before real users.
- Capture and sync should be best-effort; inference must not fail because data capture failed.

## Coding Style

- Use TypeScript only in consumers; this repo is Python-first.
- Use clear module boundaries instead of moving all scripts into one file.
- Avoid manual schema drift: contract responses must validate against `docs/ml/schema/`.
- Do not edit generated or downloaded artifacts manually.

## Reporting

- For audits, use stable IDs: `AUDIT-001`, `AUDIT-002`, etc.
- Include severity, evidence, risk, recommendation, and verification command.


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
