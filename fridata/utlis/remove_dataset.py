"""Remove all traces of a dataset from the data directory."""

import shutil
from pathlib import Path

from fridata.config import Config
from fridata.utlis.logging import logger


def remove_dataset(name: str, config: Config) -> None:
    """Remove all traces of a dataset (datasets, distograms, embeddings, etc.).

    Args:
        name: Dataset name (e.g. AFDB-subset--20250609_1333).
        config: Config instance with data_path and separator.
    """
    base_dir = Path(config.data_path)
    sep = config.separator

    logger.info(f"Removing all traces of dataset: {name}")

    # Remove from datasets/
    datasets_path = base_dir / "datasets" / name
    if datasets_path.is_dir():
        logger.info(f"  Removing: {datasets_path}")
        shutil.rmtree(datasets_path)

    # Remove from distograms/
    distograms_path = base_dir / "distograms" / name
    if distograms_path.is_dir():
        logger.info(f"  Removing: {distograms_path}")
        shutil.rmtree(distograms_path)

    # Remove from embeddings/
    embeddings_path = base_dir / "embeddings" / name
    if embeddings_path.is_dir():
        logger.info(f"  Removing: {embeddings_path}")
        shutil.rmtree(embeddings_path)

    # Remove from coordinates/
    coordinates_path = base_dir / "coordinates" / name
    if coordinates_path.exists():
        logger.info(f"  Removing coordinate files: {coordinates_path}")
        if coordinates_path.is_dir():
            shutil.rmtree(coordinates_path)
        else:
            coordinates_path.unlink()

    # Remove from sequences/
    sequences_path = base_dir / "sequences" / f"{name}_ca.fasta"
    if sequences_path.is_file():
        logger.info(f"  Removing: {sequences_path}")
        sequences_path.unlink()

    # Remove from structures/
    # PREFIX = first segment when splitting by separator
    # DATE = last segment when splitting by separator
    parts = name.split(sep)
    if parts:
        prefix = parts[0]
        date = parts[-1]
        structures_base = base_dir / "structures" / prefix
        if structures_base.is_dir():
            for parent in structures_base.iterdir():
                if parent.is_dir():
                    date_dir = parent / date
                    if date_dir.is_dir():
                        logger.info(f"  Removing: {date_dir}")
                        shutil.rmtree(date_dir)

    logger.info("Done.")
