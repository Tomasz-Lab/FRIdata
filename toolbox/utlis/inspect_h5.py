"""Inspect HDF5 files and display content in vi editor."""

import subprocess
import tempfile
from pathlib import Path

import h5py

from toolbox.models.manage_dataset.utils import read_all_pdbs_from_h5
from toolbox.utlis.logging import logger


def _inspect_in_vi(content: str) -> None:
    """Write content to temp file and open in vi."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        subprocess.run(["vi", tmp_path], check=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def inspect_h5(path: Path, mode: str = "structure") -> None:
    """Read h5 file and display in vi editor.

    Args:
        path: Path to the HDF5 file.
        mode: Display mode - 'structure' (groups/datasets/shapes), 'content'
            (PDB text via read_all_pdbs_from_h5), or 'keys' (protein codes only).
    """
    if not path.exists():
        logger.error(f"File not found: {path}")
        return

    lines = []
    if mode == "structure":
        with h5py.File(path, "r") as hf:

            def walk(name, obj, prefix=""):
                indent = "  " * prefix.count("/")
                if isinstance(obj, h5py.Group):
                    lines.append(f"{indent}{name}/ (Group)")
                    for k in obj.keys():
                        walk(k, obj[k], f"{prefix}/{k}")
                elif isinstance(obj, h5py.Dataset):
                    lines.append(
                        f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}"
                    )

            for k in hf.keys():
                walk(k, hf[k], k)

        content = "\n".join(lines)
    elif mode == "content":
        pdbs = read_all_pdbs_from_h5(str(path))
        if pdbs is None:
            logger.error("Failed to read h5 file")
            return
        for code, pdb_text in pdbs.items():
            lines.append(f"=== {code} ===")
            lines.append(pdb_text)
            lines.append("")
        content = "\n".join(lines)
    else:
        with h5py.File(path, "r") as hf:
            keys = []
            if "files" in hf:
                for ds_name in hf["files"].keys():
                    keys.extend(ds_name.split(";"))
            else:
                keys = list(hf.keys())
            content = "\n".join(sorted(keys))

    _inspect_in_vi(content)
