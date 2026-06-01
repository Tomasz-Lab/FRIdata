from pathlib import Path

from toolbox.scripts.create_archive.numeric import create_numeric_archive
from toolbox.scripts.create_archive.structures import create_structures_archive
from toolbox.scripts.create_archive.utils import (
    archive_output_dir,
    archive_timestamp,
    normalize_archive_types,
)
from toolbox.utlis.logging import logger


def create_archive(
    structures_dataset: "StructuresDataset",
    keep_shard_zips: bool = False,
    archive_types: str | list[str] | set[str] | None = None,
) -> dict[str, Path]:
    selected_types = normalize_archive_types(archive_types)
    output_dir = archive_output_dir(structures_dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = archive_timestamp()
    outputs: dict[str, Path] = {}

    logger.info(
        "Creating archive outputs for %s: %s",
        structures_dataset.dataset_dir_name(),
        ", ".join(sorted(selected_types)),
    )

    if "structures" in selected_types:
        path = create_structures_archive(
            structures_dataset,
            output_dir,
            timestamp,
            keep_shard_zips=keep_shard_zips,
        )
        if path is not None:
            outputs["structures"] = path

    for archive_type in ("embeddings", "distograms", "coordinates"):
        if archive_type not in selected_types:
            continue
        path = create_numeric_archive(
            structures_dataset,
            output_dir,
            timestamp,
            archive_type,
        )
        if path is not None:
            outputs[archive_type] = path

    if not outputs:
        logger.error("No archive outputs were created")

    return outputs
