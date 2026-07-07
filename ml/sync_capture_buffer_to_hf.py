#!/usr/bin/env python3
"""Sync the consented triage capture buffer to a private Hugging Face dataset."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_BUFFER_DIR = Path("data/bronze/triage-captures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buffer-dir",
        type=Path,
        default=Path(os.getenv("PETCARE_CAPTURE_BUFFER_DIR", DEFAULT_BUFFER_DIR)),
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_CAPTURE_DATASET_REPO"),
        help="Hugging Face dataset repo id, e.g. username/petcare-triage-captures-private.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the dataset repo as private if it does not already exist.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without uploading.")
    parser.add_argument(
        "--commit-message",
        default="Sync PetCare consented triage captures",
    )
    return parser.parse_args()


def require_hf_cli() -> None:
    if shutil.which("hf") is None:
        raise SystemExit("Missing `hf` CLI. Install with: curl -LsSf https://hf.co/cli/install.sh | bash -s")


def validate_inputs(buffer_dir: Path, repo_id: str | None) -> str:
    if not repo_id:
        raise SystemExit("Set --repo-id or HF_CAPTURE_DATASET_REPO.")
    if not buffer_dir.is_dir():
        raise SystemExit(f"Capture buffer does not exist: {buffer_dir}")
    metadata = buffer_dir / "metadata.jsonl"
    if not metadata.is_file():
        raise SystemExit(f"Capture metadata is missing: {metadata}")
    return repo_id


def upload_command(buffer_dir: Path, repo_id: str, commit_message: str) -> list[str]:
    return [
        "hf",
        "upload",
        repo_id,
        str(buffer_dir),
        ".",
        "--repo-type",
        "dataset",
        "--private",
        "--include",
        "metadata.jsonl",
        "--include",
        "README.md",
        "--include",
        "images/**",
        "--commit-message",
        commit_message,
    ]


def create_repo_command(repo_id: str) -> list[str]:
    return ["hf", "repos", "create", repo_id, "--type", "dataset", "--private", "--exist-ok"]


def print_command(command: list[str]) -> None:
    print(" ".join(command))


def main() -> int:
    args = parse_args()
    repo_id = validate_inputs(args.buffer_dir, args.repo_id)
    require_hf_cli()

    commands: list[list[str]] = []
    if args.create_repo:
        commands.append(create_repo_command(repo_id))
    commands.append(upload_command(args.buffer_dir, repo_id, args.commit_message))

    if args.dry_run:
        for command in commands:
            print_command(command)
        return 0

    for command in commands:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
