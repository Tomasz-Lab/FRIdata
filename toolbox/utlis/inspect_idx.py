"""Inspect idx (JSON) files and display in vi editor."""

import json
from pathlib import Path

from toolbox.utlis.inspect_h5 import _inspect_in_vi
from toolbox.utlis.logging import logger


def inspect_idx(path: Path) -> None:
    """Read idx file (JSON), format as pretty JSON, display in vi."""
    if not path.exists():
        logger.error(f"File not found: {path}")
        return
    with path.open() as f:
        data = json.load(f)
    content = json.dumps(data, indent=2)
    _inspect_in_vi(content)
