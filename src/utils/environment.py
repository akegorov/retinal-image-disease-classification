"""Where a run happened: git state, library versions, device and OS."""

import platform
import subprocess
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision

from src.utils.device import get_device
from src.utils.paths import PROJECT_ROOT


def _git(*args: str) -> str:
    """Output of a git command in the project root, "unknown" if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def environment() -> dict[str, Any]:
    """Git commit and dirty flag, library versions, device and OS."""
    status = _git("status", "--porcelain")
    return {
        "env/git_commit": _git("rev-parse", "HEAD"),
        "env/git_dirty": "unknown" if status == "unknown" else bool(status),
        "env/python": platform.python_version(),
        "env/torch": torch.__version__,
        "env/torchvision": torchvision.__version__,
        "env/numpy": np.__version__,
        "env/sklearn": sklearn.__version__,
        "env/device": str(get_device()),
        "env/os": platform.platform(),
    }
