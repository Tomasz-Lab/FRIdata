import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from toolbox.models.manage_dataset.index.handle_index import read_index


ARCHIVE_TYPES = frozenset(
    {
        "structures",
        "embeddings",
        "distograms",
        "coordinates",
    }
)


def normalize_archive_types(raw_types: str | Iterable[str] | None) -> set[str]:
    if raw_types is None:
        return set(ARCHIVE_TYPES)

    if isinstance(raw_types, str):
        requested = {item.strip().lower() for item in raw_types.split(",") if item.strip()}
    else:
        requested = {str(item).strip().lower() for item in raw_types if str(item).strip()}

    if not requested or "all" in requested:
        return set(ARCHIVE_TYPES)

    invalid = requested - ARCHIVE_TYPES
    if invalid:
        valid = ", ".join(["all", *sorted(ARCHIVE_TYPES)])
        raise ValueError(
            f"Invalid archive type(s): {', '.join(sorted(invalid))}. "
            f"Valid values are: {valid}"
        )

    return requested


def archive_output_dir(structures_dataset: "StructuresDataset") -> Path:
    return (
        Path(structures_dataset.config.data_path)
        / "archives"
        / structures_dataset.dataset_dir_name()
    )


def archive_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def archive_file_path(
    output_dir: Path,
    prefix: str,
    dataset_name: str,
    timestamp: str,
    suffix: str,
) -> Path:
    return output_dir / f"archive_{prefix}_{dataset_name}_{timestamp}.{suffix}"


def read_reversed_index(
    structures_dataset: "StructuresDataset",
    index_name: str,
) -> dict[str, list[str]]:
    data_path = structures_dataset.config.data_path
    index_path = Path(structures_dataset.dataset_path()) / f"{index_name}_reversed.idx"
    raw_index = read_index(index_path, data_path)
    normalized: dict[str, list[str]] = {}

    for h5_file, raw_ids in raw_index.items():
        if isinstance(raw_ids, list):
            ids = [str(item) for item in raw_ids]
        elif raw_ids is None:
            ids = []
        else:
            ids = [str(raw_ids)]
        normalized[h5_file] = ids

    return normalized


def protein_array_key(index: int) -> str:
    return f"arr_{index:08d}"


def save_npz_with_protein_ids(path: Path, arrays: list[tuple[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "__protein_ids__": np.array([protein_id for protein_id, _ in arrays], dtype=str)
    }
    for index, (_, array) in enumerate(arrays):
        payload[protein_array_key(index)] = array
    np.savez(path, **payload)
