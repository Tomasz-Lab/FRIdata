"""Inspect HDF5 files and display content in vi editor."""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np

from toolbox.models.manage_dataset.utils import read_pdbs_from_h5
from toolbox.utlis.logging import logger

FULL_STATS_MAX_ELEMENTS = 50_000
PREVIEW_SIDE = 4


def _corner_preview(ds: h5py.Dataset, side: int = PREVIEW_SIDE) -> np.ndarray:
    shape = ds.shape
    if len(shape) == 0:
        return np.asarray(ds[()])
    if len(shape) == 1:
        n = min(side, max(1, int(shape[0])))
        return np.asarray(ds[:n])
    r = min(side, max(1, int(shape[0])))
    c = min(side, max(1, int(shape[1])))
    idx: tuple = (slice(0, r), slice(0, c)) + tuple(0 for _ in range(2, len(shape)))
    return np.asarray(ds[idx])


def _summarize_numeric_dataset(ds: h5py.Dataset, title: str) -> list[str]:
    lines = [f"=== {title} ===", f"shape: {tuple(ds.shape)}, dtype: {ds.dtype}"]
    if ds.compression is not None:
        lines.append(f"compression: {ds.compression}")

    size = getattr(ds, "size", None)
    n_elem = int(size) if size is not None else int(np.prod(ds.shape))

    if n_elem == 0:
        lines.append("preview: (empty)")
        return lines

    pv = _corner_preview(ds)
    lines.append("preview:")
    lines.append(np.array2string(pv, precision=6, max_line_width=120))

    if n_elem <= FULL_STATS_MAX_ELEMENTS:
        data = ds[:]
        arr = np.asarray(data, dtype=np.float64)
        lines.append(
            "min: %.6g, max: %.6g, mean: %.6g, nan_count: %d"
            % (
                float(np.nanmin(arr)),
                float(np.nanmax(arr)),
                float(np.nanmean(arr)),
                int(np.sum(np.isnan(arr))),
            )
        )
    else:
        lines.append(
            f"full-volume statistics omitted (elements={n_elem} exceeds "
            f"{FULL_STATS_MAX_ELEMENTS}); preview slice only"
        )

    return lines


def render_inspect_h5_content(
    path: Path,
    mode: str,
    names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Build the UTF-8 text buffer that inspect_h5 would show in ``vi``.

    Args:
        path: Path to HDF5 file.
        mode: ``structure``, ``keys``, ``content``, ``distogram``, ``coordinates``, ``embedding``.
        names: Keys to include for ``content`` / ``distogram`` / ``coordinates`` / ``embedding``;
               order follows this sequence when supplied.

    Returns:
        Summary string or ``None`` if the path is missing or PDB read failed.
    """
    if not path.exists():
        logger.error(f"File not found: {path}")
        return None

    lines: list[str] = []

    if mode == "structure":
        with h5py.File(path, "r") as hf:

            def walk(
                name: str, obj: Union[h5py.Dataset, h5py.Group], prefix: str = ""
            ) -> None:
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

        return "\n".join(lines)

    if mode == "keys":
        with h5py.File(path, "r") as hf:
            keys: list[str] = []
            if "files" in hf:
                for ds_name in hf["files"].keys():
                    keys.extend(ds_name.split(";"))
            else:
                keys = list(hf.keys())
            return "\n".join(sorted(keys))

    key_order = list(names) if names else None

    if mode == "content":
        pdbs = read_pdbs_from_h5(str(path), list(names) if names else None)
        if pdbs is None:
            logger.error("Failed to read h5 file")
            return None
        if names:
            missing = set(names) - set(pdbs.keys())
            if missing:
                logger.warning(
                    "PDB keys not found in h5 (skipped): %s",
                    ", ".join(sorted(missing)),
                )
            for code in names:
                if code not in pdbs:
                    continue
                lines.append(f"=== {code} ===")
                lines.append(pdbs[code])
                lines.append("")
        else:
            for code, pdb_text in pdbs.items():
                lines.append(f"=== {code} ===")
                lines.append(pdb_text)
                lines.append("")
        return "\n".join(lines)

    with h5py.File(path, "r") as hf:
        if key_order is None:
            keys_to_walk = sorted(hf.keys(), key=str)
        else:
            keys_to_walk = list(key_order)
            missing = [k for k in keys_to_walk if k not in hf]
            if missing:
                logger.warning(
                    "H5 keys not found (skipped): %s",
                    ", ".join(sorted(missing)),
                )

        if mode == "distogram":
            for k in keys_to_walk:
                if k not in hf:
                    continue
                obj = hf[k]
                if not isinstance(obj, h5py.Group) or "distogram" not in obj:
                    logger.warning(
                        "inspect_h5: group %r has no 'distogram' dataset; skipped",
                        k,
                    )
                    continue
                lines.extend(_summarize_numeric_dataset(obj["distogram"], k))
                lines.append("")
        elif mode == "coordinates":
            for k in keys_to_walk:
                if k not in hf:
                    continue
                obj = hf[k]
                if not isinstance(obj, h5py.Group) or "coords" not in obj:
                    logger.warning(
                        "inspect_h5: group %r has no 'coords' dataset; skipped",
                        k,
                    )
                    continue
                lines.extend(_summarize_numeric_dataset(obj["coords"], k))
                lines.append("")
        elif mode == "embedding":
            for k in keys_to_walk:
                if k not in hf:
                    continue
                obj = hf[k]
                if not isinstance(obj, h5py.Dataset):
                    logger.warning(
                        "inspect_h5: key %r is not a top-level dataset; skipped",
                        k,
                    )
                    continue
                lines.extend(_summarize_numeric_dataset(obj, k))
                lines.append("")
        else:
            logger.error(f"Unsupported inspect mode: {mode}")
            return None

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


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


def inspect_h5(
    path: Path,
    mode: str = "structure",
    names: Optional[Sequence[str]] = None,
) -> None:
    """Read h5 file and display in vi editor.

    Args:
        path: Path to the HDF5 file.
        mode: ``structure`` (tree), ``keys``, ``content`` (PDB shard),
            ``distogram``, ``coordinates``, or ``embedding`` (numeric summaries).
        names: For ``content`` / ``distogram`` / ``coordinates`` / ``embedding``,
            optional exact keys; output order follows this list when given.
    """
    content = render_inspect_h5_content(path, mode, names)
    if content is None:
        return
    _inspect_in_vi(content)
