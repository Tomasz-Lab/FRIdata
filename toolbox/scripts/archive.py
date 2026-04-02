import hashlib
import zipfile
from pathlib import Path

from dask.distributed import as_completed
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from toolbox.models.manage_dataset.index.handle_index import read_index
from toolbox.models.manage_dataset.utils import read_all_pdbs_from_h5
from toolbox.utlis.logging import logger


def _shard_zip_name(h5_file: str, data_path_str: str) -> str:
    """Stable unique .zip basename per H5 shard (avoids .../N/pdbs.h5 → pdbs.zip collisions)."""
    h5_posix = Path(h5_file).as_posix()
    root = Path(data_path_str).as_posix().rstrip("/")
    if root and h5_posix.startswith(root + "/"):
        rel = h5_posix.removeprefix(root + "/")
    else:
        rel = h5_posix
    stem = rel.removesuffix(".h5").replace("/", "__")
    if len(stem) > 200:
        stem = hashlib.sha256(rel.encode()).hexdigest()[:24]
    return f"{stem}.zip"


def process_h5_file(h5_file, data_path_str, output_dir):
    h5_path = Path(h5_file)
    prots = read_all_pdbs_from_h5(str(h5_path))
    archive_name = _shard_zip_name(h5_file, data_path_str)
    archive_path = Path(output_dir) / archive_name

    with zipfile.ZipFile(archive_path, "w") as zipf:
        for p, pdb_file_content in prots.items():
            code = p.removesuffix(".pdb")
            zipf.writestr(f"{code}.pdb", pdb_file_content)

    return str(archive_path)


def create_archive(structures_dataset: "StructuresDataset"):
    data_path = structures_dataset.config.data_path
    dataset_path = structures_dataset.dataset_path()
    proteins_index = read_index(
        Path(dataset_path) / "dataset_reversed.idx", data_path
    )
    output_dir = Path(data_path) / "archives" / structures_dataset.dataset_dir_name()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = structures_dataset._client

    futures = []
    for h5_file in proteins_index.keys():
        future = client.submit(
            process_h5_file, h5_file, data_path, output_dir
        )
        futures.append(future)

    n = len(futures)
    logger.info("Writing %s PDB shard zip(s) under %s", n, output_dir)

    with logging_redirect_tqdm():
        with tqdm(total=n, desc="H5 shards → zip", unit="h5") as pbar:
            for fut in as_completed(futures):
                fut.result()
                pbar.update(1)
