from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np

from toolbox.scripts.create_archive.utils import (
    archive_file_path,
    read_reversed_index,
    save_npz_with_protein_ids,
)
from toolbox.utlis.logging import logger


ArrayReader = Callable[[h5py.File, str], Optional[np.ndarray]]


@dataclass(frozen=True)
class NumericArchiveSpec:
    archive_type: str
    index_name: str
    output_prefix: str
    reader: ArrayReader


def _candidate_ids(protein_id: str) -> tuple[str, ...]:
    stripped = protein_id.strip()
    stem = stripped.removesuffix(".pdb")
    if stripped == stem:
        return (stripped, f"{stem}.pdb")
    return (stripped, stem)


def _read_root_dataset(hf: h5py.File, protein_id: str) -> Optional[np.ndarray]:
    for candidate in _candidate_ids(protein_id):
        if candidate in hf and isinstance(hf[candidate], h5py.Dataset):
            return hf[candidate][:]
    return None


def _read_group_dataset(
    hf: h5py.File,
    protein_id: str,
    dataset_name: str,
) -> Optional[np.ndarray]:
    for candidate in _candidate_ids(protein_id):
        if candidate in hf and dataset_name in hf[candidate]:
            return hf[candidate][dataset_name][:]
    return None


def _read_coordinates(hf: h5py.File, protein_id: str) -> Optional[np.ndarray]:
    return _read_group_dataset(hf, protein_id, "coords")


def _read_distogram(hf: h5py.File, protein_id: str) -> Optional[np.ndarray]:
    return _read_group_dataset(hf, protein_id, "distogram")


NUMERIC_ARCHIVE_SPECS: dict[str, NumericArchiveSpec] = {
    "embeddings": NumericArchiveSpec(
        archive_type="embeddings",
        index_name="embeddings",
        output_prefix="embeddings",
        reader=_read_root_dataset,
    ),
    "distograms": NumericArchiveSpec(
        archive_type="distograms",
        index_name="distograms",
        output_prefix="distograms",
        reader=_read_distogram,
    ),
    "coordinates": NumericArchiveSpec(
        archive_type="coordinates",
        index_name="coordinates",
        output_prefix="coordinates",
        reader=_read_coordinates,
    ),
}


def create_numeric_archive(
    structures_dataset: "StructuresDataset",
    output_dir: Path,
    timestamp: str,
    archive_type: str,
) -> Optional[Path]:
    spec = NUMERIC_ARCHIVE_SPECS[archive_type]
    reversed_index = read_reversed_index(structures_dataset, spec.index_name)
    arrays: list[tuple[str, np.ndarray]] = []
    seen_ids: set[str] = set()

    for h5_file, protein_ids in sorted(reversed_index.items()):
        if not protein_ids:
            logger.warning(
                "No protein IDs listed for shard %s in %s_reversed.idx; skipping",
                h5_file,
                spec.index_name,
            )
            continue

        try:
            with h5py.File(h5_file, "r") as hf:
                for protein_id in protein_ids:
                    if protein_id in seen_ids:
                        logger.warning(
                            "Duplicate %s entry for %s; keeping first value",
                            spec.archive_type,
                            protein_id,
                        )
                        continue
                    array = spec.reader(hf, protein_id)
                    if array is None:
                        logger.warning(
                            "%s H5 %s: protein %s not found; skipping",
                            spec.archive_type,
                            h5_file,
                            protein_id,
                        )
                        continue
                    arrays.append((protein_id, array))
                    seen_ids.add(protein_id)
        except OSError as exc:
            logger.error("Could not read %s H5 %s: %s", spec.archive_type, h5_file, exc)

    if not arrays:
        logger.error(
            "No %s arrays exported. Check %s_reversed.idx and referenced H5 files.",
            spec.archive_type,
            spec.index_name,
        )
        return None

    output_path = archive_file_path(
        output_dir,
        spec.output_prefix,
        structures_dataset.dataset_dir_name(),
        timestamp,
        "npz",
    )
    save_npz_with_protein_ids(output_path, arrays)
    logger.info("Wrote %s archive %s", spec.archive_type, output_path)
    return output_path
