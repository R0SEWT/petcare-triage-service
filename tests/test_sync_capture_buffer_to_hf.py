from pathlib import Path

from ml.sync_capture_buffer_to_hf import create_repo_command, upload_command


def test_upload_command_targets_private_dataset_repo(tmp_path: Path):
    command = upload_command(tmp_path, "user/private-captures", "sync captures")

    assert command[:5] == ["hf", "upload", "user/private-captures", str(tmp_path), "."]
    assert "--repo-type" in command
    assert "dataset" in command
    assert "--private" in command
    assert "images/**" in command
    assert "metadata.jsonl" in command


def test_create_repo_command_is_private_dataset():
    assert create_repo_command("user/private-captures") == [
        "hf",
        "repos",
        "create",
        "user/private-captures",
        "--type",
        "dataset",
        "--private",
        "--exist-ok",
    ]
