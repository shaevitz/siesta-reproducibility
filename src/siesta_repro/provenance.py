"""Small helpers for hash-bound run receipts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from .records import sha256_file


def git_revision(directory: str | Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_receipt(path: str | Path, *, operation: str, inputs: Iterable[str | Path], parameters: dict[str, Any], result: dict[str, Any]) -> None:
    destination = Path(path)
    payload = {
        "operation": operation,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(Path(__file__).resolve().parents[2]),
        "inputs": [{"path": str(Path(item).resolve()), "sha256": sha256_file(item)} for item in inputs],
        "parameters": parameters,
        "result": result,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
