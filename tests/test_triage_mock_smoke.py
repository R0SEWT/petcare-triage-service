import subprocess
import sys
from pathlib import Path


def test_triage_mock_contract_smoke():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "smoke_test.py"],
        cwd=repo / "services" / "triage-mock",
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

