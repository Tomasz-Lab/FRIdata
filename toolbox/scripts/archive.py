from toolbox.models.manage_dataset.index.handle_index import read_index
import datetime
import os
import zipfile
from pathlib import Path

from dask.distributed import as_completed
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from toolbox.models.manage_dataset.utils import read_all_pdbs_from_h5
from toolbox.utlis.logging import logger


def process_h5_file(h5_file, dataset_path, output_dir):
    full_h5_file_path = Path(dataset_path) / h5_file
    prots = read_all_pdbs_from_h5(full_h5_file_path)
    archive_name = os.path.basename(h5_file).replace(".h5", ".zip")
    archive_path = Path(output_dir) / archive_name

    with zipfile.ZipFile(archive_path, "w") as zipf:
        for p, pdb_file_content in prots.items():
            code = p.removesuffix(".pdb")
            zipf.writestr(f"{code}.pdb", pdb_file_content)

    os.system(f"tar -czf {str(archive_path)}.tgz {str(archive_path)}")

    return str(archive_path)


def create_archive(structures_dataset: "StructuresDataset"):
    dataset_path = structures_dataset.dataset_path()
    proteins_index = read_index(Path(dataset_path) / "dataset_reversed.idx", structures_dataset.config.data_path)
    output_dir = Path(dataset_path) / "archives"
    output_dir.mkdir(exist_ok=True)

    client = structures_dataset._client

    futures = []
    for h5_file in proteins_index.keys():
        future = client.submit(process_h5_file, h5_file, dataset_path, output_dir)
        futures.append(future)

    n = len(futures)
    logger.info("Building combined PDB archive from %s H5 shard(s)", n)

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    final_archive_name = f"archive_pdb_{structures_dataset.dataset_dir_name()}_{current_time}.zip"
    final_archive_path = Path.cwd() / final_archive_name

    with zipfile.ZipFile(final_archive_path, "w") as final_zip:
        with logging_redirect_tqdm():
            with tqdm(total=n, desc="H5 shards → final zip", unit="h5") as pbar:
                i = 0
                for fut in as_completed(futures):
                    archive_path = fut.result()
                    with open(archive_path, "rb") as f:
                        archive_data = f.read()
                    final_zip.writestr(f"{i}.zip", archive_data)
                    i += 1
                    pbar.update(1)
