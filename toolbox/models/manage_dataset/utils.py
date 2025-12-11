import asyncio
import os
import shutil
import tempfile
import time
import traceback
import zlib

from io import BytesIO, StringIO
from itertools import islice
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed as futures_as_completed

import biotite.database
import biotite.database.rcsb
import biotite.database.afdb
import dask
import h5py
import numpy as np
from dask.distributed import as_completed as dask_as_completed, worker_client
from foldcomp import foldcomp
from foldcomp.setup import download

from toolbox.models.manage_dataset.compress_experiment.exp import (
    compress_and_save_h5_combined_lzf_shuffle,
    compress_and_save_h5_individual,
    compress_and_save_h5_individual_lzf,
    compress_and_save_h5_combined,
    compress_and_save_h5_combined_lzf,
    compress_and_save_h5_individual_lzf_shuffle,
)
from toolbox.models.utils.cif2pdb import cif_to_pdb, binary_cif_to_pdb

from toolbox.utlis.logging import logger


def foldcomp_download(db: str, output_dir: str):
    logger.info(f"Foldcomp downloading db: {db} to {output_dir}")
    download_chunks = 16
    for i in ["", ".index", ".dbtype", ".lookup", ".source"]:
        asyncio.run(
            download(
                f"https://foldcomp.steineggerlab.workers.dev/{db}{i}",
                f"{output_dir}/{db}{i}",
                chunks=download_chunks,
            )
        )


def chunk(data, size):
    it = iter(data)
    while True:
        chunk_data = tuple(islice(it, size))
        if not chunk_data:
            break
        yield chunk_data


def retrieve_cif(pdb: str) -> Tuple[Optional[str], str]:
    retry_num: int = 0
    cif_file: Optional[str] = None

    while retry_num <= 3 and cif_file is None:
        if retry_num > 0:
            logger.debug(f"Retrying downloading {pdb} {retry_num}")

        try:
            cif_file_io: StringIO = biotite.database.rcsb.fetch(pdb, "cif")
            cif_file: str = cif_file_io.getvalue()
        except Exception:
            cif_file = None

        if not cif_file:
            retry_num += 1

    if retry_num > 3:
        logger.warning(f"Failed retrying {pdb}")
        return None, pdb

    return cif_file, pdb


def cif_to_pdbs(input_data) -> Tuple[Dict[str, str], Tuple[str, Optional[str]]]:
    cif_file, pdb = input_data
    if cif_file is None:
        return {}, (pdb, None)

    converted = {}
    try:
        converted = cif_to_pdb(cif_file, pdb)
    except Exception as e:
        traceback.print_exc()
        logger.error(e)
        logger.error(f"ERROR: {pdb}, problem with cif conversion.")
        return {}, (pdb, None)

    if converted is None:
        return {}, (pdb, None)

    return converted, (pdb, cif_file)


def binary_cif_to_pdbs(input_data) -> Tuple[Dict[str, str], Tuple[str, BytesIO | None]]:
    binary_file_bytes_io, pdb = input_data
    if binary_file_bytes_io is None:
        return {}, (pdb, None)
    converted = {}
    try:
        converted = binary_cif_to_pdb(binary_file_bytes_io, pdb)
    except Exception as e:
        traceback.print_exc()
        logger.error(e)
        logger.error(f"Error in converting cif {pdb}")

    return converted, (pdb, binary_file_bytes_io)


def retrieve_binary_cif(pdb: str) -> tuple[BytesIO | None, str]:
    retry_num: int = 0
    binary_file_bytes_io: Optional[BytesIO] = None

    while retry_num <= 3 and binary_file_bytes_io is None:

        if retry_num > 0:
            logger.debug(f"Retrying downloading {pdb} {retry_num}")

        try:
            binary_file_bytes_io: BytesIO = biotite.database.rcsb.fetch(pdb, "bcif")
        except Exception:
            binary_file_bytes_io = None

        if not binary_file_bytes_io:
            retry_num += 1

    if retry_num > 3:
        logger.warning(f"Failed retrying {pdb}")
        return None, pdb

    return binary_file_bytes_io, pdb


def process_future(future: Tuple[Dict[str, str], Tuple[str, str]]):
    # Directly use future since it's a tuple containing the results
    chains: Dict[str, str] = future[0]
    cif_file: Tuple[str, str] = future[1]

    return chains.keys(), chains.values(), cif_file


def aggregate_results(
    protein_pdbs_with_cif: List[Tuple[Dict[str, str], Tuple[str, str]]],
    download_start_time: float,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    end_time = time.time()

    logger.debug(f"Download time: {format_time(end_time - download_start_time)}")

    all_res_pdbs = []
    all_contents = []
    cif_files = []

    for prot, cif_file in protein_pdbs_with_cif:
        all_res_pdbs.extend(prot.keys())
        all_contents.extend(prot.values())
        cif_files.append(cif_file)

    return all_res_pdbs, all_contents, cif_files


# def create_zip_archive(path_for_batch: Path, results):
#     zip_path = path_for_batch / 'cif_files.zip'
#     cif_files = results[2]
#     with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for cif_file_name, cif_str in cif_files:
#             cif_file_name = cif_file_name if cif_file_name.endswith('.cif') else cif_file_name + '.cif'
#             zipf.writestr(cif_file_name, cif_str)
#     return str(zip_path)
#
#
# def create_pdb_zip_archive(path_for_batch: Path, results):
#     zip_path = path_for_batch / 'pdb_files.zip'
#     all_res_pdbs = results[0]
#     all_contents = results[1]
#     with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for pdb_file_name, pdb_str in zip(all_res_pdbs, all_contents):
#             pdb_file_name = pdb_file_name if pdb_file_name.endswith('.pdb') else pdb_file_name + '.pdb'
#             zipf.writestr(pdb_file_name, pdb_str)
#     return str(zip_path)


def compress_and_save_h5(
    path_for_batch: Path, results: Tuple[List[str], List[str], List[str]]
):
    start_time = time.time()
    pdbs_file = path_for_batch / "pdbs.h5"
    all_res_pdbs = results[0]
    all_contents = results[1]
    if len(all_contents) == 0 or len(all_res_pdbs) == 0:
        logger.info("No files to save")
        return None
    if len(all_res_pdbs) != len(all_res_pdbs):
        logger.warning("Wrong length of names and pdb contents")
        return None
    with h5py.File(pdbs_file, "w") as hf:
        files_group = hf.create_group("files")
        files_together = zlib.compress("|".join(all_contents).encode("utf-8"))
        pdbs_content = np.frombuffer(files_together, dtype=np.uint8)
        files_group.create_dataset(name=";".join(all_res_pdbs), data=pdbs_content)
    end_time = time.time()
    total_time = end_time - start_time
    logger.debug(f"Compress time: {format_time(total_time)}")
    return str(pdbs_file)


def compress_and_save_experiment(
    path_for_batch: Path, results: Tuple[List[str], List[str], List[str]]
):
    fs = [
        compress_and_save_h5_individual,
        compress_and_save_h5_individual_lzf,
        compress_and_save_h5_individual_lzf_shuffle,
        compress_and_save_h5_combined,
        compress_and_save_h5_combined_lzf,
        compress_and_save_h5_combined_lzf_shuffle,
        compress_and_save_h5,
    ]

    descriptions = [
        "individual gzip",
        "individual lzf",
        "individual lzf shuffle",
        "combined gzip",
        "combined lzf",
        "combined lzf shuffle",
        "combined zlib",
    ]

    inputs = list(results)

    def get_file_size_mb(file_path):
        try:
            size_in_bytes = os.path.getsize(file_path)
            size_in_mb = size_in_bytes / (1024 * 1024)  # Convert bytes to megabytes
            return round(size_in_mb, 2)
        except Exception:
            return None

    for f, desc in zip(fs, descriptions):
        logger.debug(desc)
        path = f(path_for_batch, inputs)
        logger.debug(f"{path} {get_file_size_mb(path)} MB")


def retrieve_pdb_chunk_to_h5(
    path_for_batch: Path,
    pdb_ids: Iterable[str],
    is_binary: bool,
    workers: List[str] = None,
) -> Tuple[List[str], str]:
    with worker_client() as client:
        # start_time = time.time()

        pdb_futures = client.map(
            retrieve_binary_cif if is_binary else retrieve_cif, pdb_ids, workers=workers
        )
        converted_pdb_futures = client.map(
            binary_cif_to_pdbs if is_binary else cif_to_pdbs,
            pdb_futures,
            workers=workers,
        )
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
        # zip_task = client.submit(create_cif_files_zip_archive, path_for_batch, aggregated, pure=False)
        # pdb_zip_task = client.submit(create_pdb_zip_archive, path_for_batch, aggregated, pure=False)

        # Compute the tasks
        pdb_ids, h5_file_path = client.gather([get_ids_task, h5_task])

        # end_time = time.time()
        # total_time = end_time - start_time
        # logger.info(f"Total processing time {path_for_batch.stem}: {format_time(total_time)}")

        return pdb_ids, h5_file_path


def fetch_one_afdb(
    uniprot_id: str,
    target_path: Path,
    max_attempts: int = 3,
    retry_delay_s: float = 1.0,
) -> tuple[str, Optional[Exception]]:
    """
    Download a single AFDB structure.
    
    Args:
        uniprot_id: UniProt ID to download
        target_path: Path where the CIF file should be saved
        max_attempts: Total attempts before giving up (default: 3)
        retry_delay_s: Delay between attempts in seconds (default: 1.0)
    
    Returns:
        Tuple of (uniprot_id, None) on success or (uniprot_id, exception) on failure
    """
    last_error: Optional[Exception] = None
    attempts = max(1, int(max_attempts))

    for attempt in range(1, attempts + 1):
        try:
            biotite.database.afdb.fetch(
                ids=[uniprot_id],
                format="cif",
                target_path=str(target_path),
                overwrite=False,
            )
            logger.debug(f"Successfully downloaded AFDB structure for ID {uniprot_id}")
            return uniprot_id, None
        except Exception as e:
            last_error = e
            if attempt < attempts:
                logger.warning(
                    f"AFDB download failed for ID {uniprot_id} (attempt {attempt}/{attempts}): {e}. "
                    f"Retrying in {retry_delay_s}s..."
                )
                time.sleep(max(0.0, float(retry_delay_s)))
            else:
                logger.error(
                    f"Failed to download AFDB structure for ID {uniprot_id} after {attempts} attempts: {e}"
                )

    return uniprot_id, last_error


def retrieve_afdb_chunk_to_h5_concurrent(
    path_for_batch: Path,
    uniprot_ids: Iterable[str],
    max_workers: int = 8,
) -> Tuple[List[str], str]:
    """
    Download AFDB structures for a chunk of UniProt IDs concurrently using ThreadPoolExecutor,
    convert to PDB format, and save to HDF5 file.
    
    Args:
        path_for_batch: Path where the HDF5 file will be saved
        uniprot_ids: Iterable of UniProt IDs to download
        max_workers: Maximum number of concurrent download threads (default: 8)
    
    Returns:
        Tuple of (list of successfully processed IDs, path to HDF5 file)
    """
    chunk_ids = list(uniprot_ids)
    if not chunk_ids:
        return [], ""
    
    temp_dir = None
    try:
        # Create temporary directory for downloading CIF files
        temp_dir = Path(tempfile.mkdtemp())
        
        logger.debug(f"Downloading {len(chunk_ids)} AFDB structures to {temp_dir} with {max_workers} workers")
        
        # Download CIF files concurrently using ThreadPoolExecutor
        successful_downloads = []
        failed_downloads = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one_afdb, uniprot_id, temp_dir): uniprot_id 
                      for uniprot_id in chunk_ids}
            
            for future in futures_as_completed(futures):
                uniprot_id = futures[future]
                try:
                    downloaded_id, error = future.result()
                    if error is None:
                        successful_downloads.append(downloaded_id)
                    else:
                        failed_downloads.append((downloaded_id, error))
                except Exception as e:
                    logger.error(f"Unexpected error processing {uniprot_id}: {e}")
                    failed_downloads.append((uniprot_id, e))
        
        if not successful_downloads:
            logger.warning(f"No AFDB structures were successfully downloaded for chunk (all {len(chunk_ids)} IDs failed)")
            return [], ""
        
        if failed_downloads:
            logger.warning(f"Failed to download {len(failed_downloads)} out of {len(chunk_ids)} IDs")
        
        logger.info(f"Successfully downloaded {len(successful_downloads)} out of {len(chunk_ids)} AFDB structures")
        
        # Process downloaded CIF files
        all_res_pdbs = []
        all_contents = []
        
        # Get all downloaded CIF files
        cif_files = list(temp_dir.glob("*.cif"))
        
        if not cif_files:
            logger.warning(f"No CIF files found in temp directory after download")
            return [], ""
        
        # Process each CIF file
        for cif_file in cif_files:
            try:
                # Read CIF file content
                with open(cif_file, 'r') as f:
                    cif_content = f.read()
                
                # Extract UniProt ID from filename (format: AF-{uniprot_id}-F1-model_v4.cif)
                filename = cif_file.stem
                # Remove AF- prefix and -F1-model_v4 suffix
                uniprot_id = filename.removesuffix("-F1-model_v4").removeprefix("AF-")
                
                # Convert CIF to PDB format
                converted = cif_to_pdb(cif_content, uniprot_id)
                
                if converted is None or len(converted) == 0:
                    logger.warning(f"Failed to convert {uniprot_id} from CIF to PDB")
                    continue
                
                # Add all chains to results
                for chain_id, pdb_content in converted.items():
                    all_res_pdbs.append(chain_id)
                    all_contents.append(pdb_content)
                    
            except Exception as e:
                logger.error(f"Error processing {cif_file}: {e}")
                traceback.print_exc()
                continue
        
        # Save to HDF5
        if len(all_res_pdbs) == 0 or len(all_contents) == 0:
            logger.warning("No PDB structures to save")
            return [], ""
        
        results = (all_res_pdbs, all_contents, [])
        h5_file_path = compress_and_save_h5(path_for_batch, results)
        
        if h5_file_path is None:
            return [], ""
        
        return all_res_pdbs, h5_file_path
        
    finally:
        # Cleanup temporary directory
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory {temp_dir}: {e}")


def retrieve_afdb_chunk_to_h5(
    path_for_batch: Path,
    uniprot_ids: Iterable[str],
    workers: List[str] = None,
) -> Tuple[List[str], str]:
    """
    Download AFDB structures for a chunk of UniProt IDs, convert to PDB format,
    and save to HDF5 file.
    
    Args:
        path_for_batch: Path where the HDF5 file will be saved
        uniprot_ids: Iterable of UniProt IDs to download
        workers: Optional list of worker names for dask
    
    Returns:
        Tuple of (list of successfully processed IDs, path to HDF5 file)
    """
    chunk_ids = list(uniprot_ids)
    if not chunk_ids:
        return [], ""
    
    temp_dir = None
    try:
        # Create temporary directory for downloading CIF files
        temp_dir = Path(tempfile.mkdtemp())
        
        # Download CIF files from AFDB one by one for better error handling
        logger.debug(f"Downloading {len(chunk_ids)} AFDB structures to {temp_dir}")
        successful_downloads = []
        failed_downloads = []
        
        for uniprot_id in chunk_ids:
            try:
                biotite.database.afdb.fetch(ids=[uniprot_id], format='cif', target_path=str(temp_dir))
                successful_downloads.append(uniprot_id)
                logger.debug(f"Successfully downloaded AFDB structure for ID {uniprot_id}")
            except Exception as e:
                logger.error(f"Failed to download AFDB structure for ID {uniprot_id}: {e}")
                failed_downloads.append(uniprot_id)
                # Continue to the next ID
                continue
        
        if not successful_downloads:
            logger.warning(f"No AFDB structures were successfully downloaded for chunk (all {len(chunk_ids)} IDs failed)")
            return [], ""
        
        if failed_downloads:
            logger.warning(f"Failed to download {len(failed_downloads)} out of {len(chunk_ids)} IDs: {failed_downloads}")
        
        logger.info(f"Successfully downloaded {len(successful_downloads)} out of {len(chunk_ids)} AFDB structures")
        
        # Process downloaded CIF files
        all_res_pdbs = []
        all_contents = []
        
        # Get all downloaded CIF files
        cif_files = list(temp_dir.glob("*.cif"))
        
        if not cif_files:
            logger.warning(f"No CIF files found in temp directory after download")
            return [], ""
        
        # Process each CIF file
        for cif_file in cif_files:
            try:
                # Read CIF file content
                with open(cif_file, 'r') as f:
                    cif_content = f.read()
                
                # Extract UniProt ID from filename (format: AF-{uniprot_id}-F1-model_v4.cif)
                filename = cif_file.stem
                # Remove AF- prefix and -F1-model_v4 suffix (same pattern as alphafold_chunk_to_h5)
                uniprot_id = filename.removesuffix("-F1-model_v4").removeprefix("AF-")
                
                # Convert CIF to PDB format
                converted = cif_to_pdb(cif_content, uniprot_id)
                
                if converted is None or len(converted) == 0:
                    logger.warning(f"Failed to convert {uniprot_id} from CIF to PDB")
                    continue
                
                # Add all chains to results
                for chain_id, pdb_content in converted.items():
                    all_res_pdbs.append(chain_id)
                    all_contents.append(pdb_content)
                    
            except Exception as e:
                logger.error(f"Error processing {cif_file}: {e}")
                traceback.print_exc()
                continue
        
        # Save to HDF5
        if len(all_res_pdbs) == 0 or len(all_contents) == 0:
            logger.warning("No PDB structures to save")
            return [], ""
        
        results = (all_res_pdbs, all_contents, [])
        h5_file_path = compress_and_save_h5(path_for_batch, results)
        
        if h5_file_path is None:
            return [], ""
        
        return all_res_pdbs, h5_file_path
        
    finally:
        # Cleanup temporary directory
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory {temp_dir}: {e}")


def mkdir_for_batches(base_path: Path, batch_count: int, offset: int = 0):
    for i in range(batch_count):
        (base_path / f"{i + offset}").mkdir(exist_ok=True, parents=True)


def alphafold_chunk_to_h5(db_path: str, structures_path_for_batch: str, ids: List[str]):
    protein_codes = []
    contents = []

    with foldcomp.open(db_path, ids=ids) as db:
        for (_, content), file_name in zip(db, ids):
            protein_id = file_name.removesuffix("-F1-model_v4.pdb").removeprefix("AF-")

            contents.append(content)
            protein_codes.append(protein_id)

    h5_file = compress_and_save_h5(
        Path(structures_path_for_batch), (protein_codes, contents, [])
    )

    if h5_file is None:
        return {}

    return {protein_code: h5_file for protein_code in protein_codes}


def read_all_pdbs_from_h5(h5_file_path: str) -> Optional[Dict[str, str]]:
    """
    Read all PDB contents from an HDF5 file.

    Args:
    h5_file_path (str): Path to the HDF5 file.

    Returns:
    Optional[Dict[str, str]]: A dictionary with PDB codes as keys and their contents as values,
                              or None if an error occurs.
    """
    h5_file_path = Path(h5_file_path)

    if not h5_file_path.exists():
        logger.error(f"Error: File {h5_file_path} does not exist.")
        return None

    try:
        with h5py.File(h5_file_path, "r") as hf:
            pdb_files = hf["files"]

            for pdb_file_names in pdb_files:
                pdb_contents_bytes = pdb_files[pdb_file_names][:].tobytes()
                decompressed = zlib.decompress(pdb_contents_bytes).decode("utf-8")
                all_pdbs = decompressed.split("|")

                all_file_names_split = pdb_file_names.split(";")

            return dict(zip(all_file_names_split, all_pdbs))
    except Exception as e:
        logger.error(f"An error occurred while reading the HDF5 file: {e}")
        return None


def read_pdbs_from_h5(
    h5_file_path: str, codes: Optional[List[str]]
) -> Optional[Dict[str, str]]:
    h5_file_path_obj = Path(h5_file_path)
    if not h5_file_path_obj.exists():
        logger.error(f"Error: File {h5_file_path_obj} does not exist.")
        return None

    codes = None if not codes else set(codes)

    try:
        with h5py.File(h5_file_path_obj, "r") as hf:
            pdb_files = hf["files"]
            result = {}

            for pdb_file_names in pdb_files:
                pdb_contents_bytes = pdb_files[pdb_file_names][:].tobytes()
                decompressed = zlib.decompress(pdb_contents_bytes).decode("utf-8")
                all_pdbs = decompressed.split("|")
                all_file_names_split = pdb_file_names.split(";")

                if codes is not None:
                    # Only include the codes that are in the 'codes' list
                    for code, pdb in zip(all_file_names_split, all_pdbs):
                        if code in codes:
                            result[code] = pdb
                else:
                    for code, pdb in zip(all_file_names_split, all_pdbs):
                        result[code] = pdb

            return result

    except Exception as e:
        logger.error(f"An error occurred while reading the HDF5 file: {e}")
        return None


def write_file(path, pdb_file_name, content):
    with open(path / pdb_file_name, "w") as f:
        f.write(content)


def pdbs_h5_to_files(h5_file_path: str):
    path = Path("./pdbs")
    path.mkdir(exist_ok=True, parents=True)

    pdbs_dict = read_all_pdbs_from_h5(h5_file_path)

    if pdbs_dict is None:
        return []

    logger.info(f"Processing {len(pdbs_dict.keys())} PDB entries")

    for i, (name, content) in enumerate(pdbs_dict.items()):
        write_file(path, f"{i}_{name}", content)


def groupby_dict_by_values(d):
    v = {}

    for key, value in d.items():
        v.setdefault(value, []).append(key)

    return v

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_whole = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds_whole > 0 or (hours == 0 and minutes == 0 and milliseconds == 0):
        parts.append(f"{seconds_whole}s")
    if milliseconds > 0:
        parts.append(f"{milliseconds}ms")
    
    return " ".join(parts)
