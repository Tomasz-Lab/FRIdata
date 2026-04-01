import os
import time
from typing import Iterable, List, Optional, Tuple, Dict, Pattern
import zipfile
import tarfile
from pathlib import Path

from dask.distributed import worker_client
from toolbox.models.manage_dataset.index.handle_index import add_new_files_to_index, create_index
from toolbox.models.utils.create_client import total_workers

from toolbox.models.manage_dataset.compute_batches import ComputeBatches
from toolbox.models.manage_dataset.utils import (
    compress_and_save_h5,
    mkdir_for_batches,
    format_time
)
from toolbox.models.utils.cif2pdb import cif_to_pdb

from toolbox.utlis.logging import logger
import re


def extract_archive(
    input_path, structures_dataset: "StructureDataset"
) -> Optional[Path]:

    if not input_path.exists():
        logger.error(f"Error: The provided path {input_path} does not exist.")
        return None

    if is_archive(input_path):
        logger.info(f"Processing archive to extract protein files: {input_path}")
        extracted_path = structures_dataset.dataset_repo_path() / "extracted_files"
        os.makedirs(extracted_path, exist_ok=True)

        if zipfile.is_zipfile(input_path):
            logger.debug("Extracting zip file")
            with zipfile.ZipFile(input_path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)
        elif tarfile.is_tarfile(input_path):
            logger.debug("Extracting tar/tar.gz file")
            with tarfile.open(
                input_path, "r:*"
            ) as tar_ref:  # 'r:*' auto-detects compression
                tar_ref.extractall(extracted_path)
        else:
            logger.error("Provided path is neither a directory nor a supported archive.")
            return None

        return extracted_path

    return input_path  # Return original path if it's not an archive


def is_archive(path):
    if os.path.isdir(path):
        return False
    # Check if the file is a zip or tar/tar.gz archive
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def build_stem_to_paths(extracted_path: Path) -> Dict[str, List[Path]]:
    """
    Collect all .pdb / .cif under extracted_path, keyed by filename stem.
    Paths are resolved to absolute for reliable opening (archive vs directory).
    """
    stem_to_paths: Dict[str, List[Path]] = {}
    base = extracted_path.resolve()
    if not base.exists():
        return stem_to_paths
    for pattern in ("*.pdb", "*.cif"):
        for p in base.rglob(pattern):
            if not p.is_file():
                continue
            resolved = p.resolve()
            stem_to_paths.setdefault(resolved.stem, []).append(resolved)
    return stem_to_paths


def _pick_path_prefer_cif(paths: List[Path]) -> Path:
    """If both .cif and .pdb exist for the same stem, prefer .cif."""
    cifs = [p for p in paths if p.suffix.lower() == ".cif"]
    if cifs:
        return sorted(cifs, key=lambda x: str(x))[0]
    return sorted(paths, key=lambda x: str(x))[0]


def _paths_at_max_af_version(
    stem_to_paths: Dict[str, List[Path]], pattern: Pattern[str]
) -> List[Path]:
    """
    Match stems with pattern groups: (fragment F, version V).
    Pick the globally highest V, then return one path per matching stem at that V
    (prefer .cif when a stem has multiple extensions).
    """
    rows: List[Tuple[int, int, str]] = []
    for stem in stem_to_paths:
        m = pattern.match(stem)
        if m:
            fragment = int(m.group(1))
            version = int(m.group(2))
            rows.append((version, fragment, stem))
    if not rows:
        return []
    max_v = max(r[0] for r in rows)
    stems_at_max = sorted({r[2] for r in rows if r[0] == max_v})
    return [_pick_path_prefer_cif(stem_to_paths[s]) for s in stems_at_max]


def resolve_id(requested_id: str, stem_to_paths: Dict[str, List[Path]]) -> List[Path]:
    """
    Resolve a requested protein id to file path(s) under input_path / extracted tree.

    1) Exact stem match (prefer .cif over .pdb).
    2) AF exact: AF-{id}-F{N}-model_v{V} — highest V, all fragments at that V.
    3) AF loose (isoform): AF-{id}-{digits}-F{N}-model_v{V} — same version rule.
    """
    if requested_id in stem_to_paths:
        return [_pick_path_prefer_cif(stem_to_paths[requested_id])]

    af_exact = re.compile(rf"^AF-{re.escape(requested_id)}-F(\d+)-model_v(\d+)$")
    found = _paths_at_max_af_version(stem_to_paths, af_exact)
    if found:
        return found

    af_loose = re.compile(rf"^AF-{re.escape(requested_id)}-\d+-F(\d+)-model_v(\d+)$")
    return _paths_at_max_af_version(stem_to_paths, af_loose)


def save_extracted_files(
    structures_dataset: "StructuresDataset",
    extracted_path: Path,
    ids: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """
    Process extracted files from archive and return missing IDs.
    
    Returns:
        List of missing IDs (processed for AFDB if needed), or None if ids was None
    """
    Path(structures_dataset.structures_path()).mkdir(exist_ok=True, parents=True)
    pdb_repo_path = structures_dataset.structures_path()

    extracted_path = Path(extracted_path)
    stem_to_paths = build_stem_to_paths(extracted_path)
    picked_by_stem = {
        stem: _pick_path_prefer_cif(paths) for stem, paths in stem_to_paths.items()
    }

    logger.debug(f"extracted files: {len(stem_to_paths)} unique stems")

    if ids is None:
        files = [str(picked_by_stem[s]) for s in sorted(picked_by_stem)]
        chunks = list(structures_dataset.chunk(files))
        missing_files = None
    else:
        logger.info(
            f"Searching for {len(ids)} ids among {len(stem_to_paths)} stems under {extracted_path}"
        )

        wanted_paths: List[str] = []
        seen_resolved: set[str] = set()
        missing_files = []

        for raw_id in ids:
            rid = raw_id.strip()
            resolved_paths = resolve_id(rid, stem_to_paths)
            if not resolved_paths:
                missing_files.append(raw_id)
                continue
            for p in resolved_paths:
                key = str(p.resolve())
                if key not in seen_resolved:
                    seen_resolved.add(key)
                    wanted_paths.append(str(p))

        logger.info(
            f"Resolved {len(wanted_paths)} file path(s), {len(missing_files)} missing "
            f"out of {len(ids)} requested ids"
        )
        ids = wanted_paths

        chunks = list(structures_dataset.chunk(ids))

    mkdir_for_batches(pdb_repo_path, len(chunks))

    new_files_index = {}

    def run(input_data, machine):
        return structures_dataset._client.submit(
            retrieve_protein_file_to_h5, *input_data, [machine], workers=[machine]
        )

    def collect(result):
        downloaded_pdbs, file_path = result
        new_files_index.update({k: file_path for k in downloaded_pdbs})

    compute_batches = ComputeBatches(
        structures_dataset._client, run, collect, "pdb_extracted_from_archive", len(chunks)
    )

    inputs = ((pdb_repo_path / f"{i}", ids_chunk) for i, ids_chunk in enumerate(chunks))

    logger.info(f"Extracting PDBs from archive {extracted_path} in {len(chunks)} batches")

    factor = 1
    factor = 15 if total_workers() > 1500 else factor
    factor = 20 if total_workers() > 2000 else factor
    compute_batches.compute(inputs, factor=factor)

    logger.info("Adding new files to index")

    no_chain_to_chain_dict = {}
    for id_with_chain in new_files_index.keys():
        # Remove last '_' and everything after it to get id_without_chain
        id_without_chain = id_with_chain.rsplit('_', 1)[0]
        no_chain_to_chain_dict[id_without_chain] = id_with_chain

    input_structures_index = {}

    for stem, picked in sorted(picked_by_stem.items(), key=lambda x: x[0]):
        file_path = picked
        id_with_chain = no_chain_to_chain_dict.get(stem, None)
        if id_with_chain is None:
            continue
        input_structures_index[id_with_chain] = file_path.name

    try:
        add_new_files_to_index(structures_dataset.dataset_index_file_path(), new_files_index, structures_dataset.config.data_path)
        create_index(structures_dataset.input_structures_index_path(), input_structures_index, structures_dataset.config.data_path)
    except Exception as e:
        logger.error(f"Failed to update index: {e}")
    
    return missing_files


def retrieve_protein_file_to_h5(
    path_for_batch: Path, pdb_ids: Iterable[str], workers: List[str] = None
) -> Tuple[List[str], str]:
    with worker_client() as client:
        start_time = time.time()

        pdb_futures = client.map(retrieve_single_file, pdb_ids, workers=workers)
        converted_pdb_futures = client.map(file_to_pdb, pdb_futures, workers=workers)
        download_start_time = time.time()
        aggregated = client.submit(
            aggregate_results,
            converted_pdb_futures,
            download_start_time,
            workers=workers,
        )

        # Create delayed tasks for H5 and ZIP creation
        h5_task = client.submit(
            compress_and_save_h5,
            path_for_batch,
            aggregated,
            pure=False,
            workers=workers,
        )
        get_ids_task = client.submit(
            lambda results: results[0], aggregated, workers=workers
        )

        # Compute the tasks
        pdb_ids, h5_file_path = client.gather([get_ids_task, h5_task])

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total processing time {path_for_batch.stem}: {format_time(total_time)}")

        return pdb_ids, h5_file_path


def retrieve_single_file(file_path):
    file_path = Path(file_path)
    file_name = file_path.stem
    file_extension = file_path.suffix
    with open(file_path, "r") as file:
        return file.read(), file_name, file_extension


def file_to_pdb(input_data):
    file_data, file_name, file_extension = input_data
    if file_extension == ".cif":
        return cif_to_pdb(file_data, file_name)
    elif file_extension == ".pdb":
        return {f"{file_name}": str(file_data)}
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def aggregate_results(
    protein_pdbs_with_cif: List[Dict[str, str]], download_start_time: float
) -> Tuple[List[str], List[str]]:
    end_time = time.time()

    logger.debug(f"Download time: {format_time(end_time - download_start_time)}")

    all_res_pdbs = []
    all_contents = []

    for prot in protein_pdbs_with_cif:
        all_res_pdbs.extend(prot.keys())
        all_contents.extend(prot.values())

    return all_res_pdbs, all_contents
