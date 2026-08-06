import pathlib
import time

from pathlib import Path
from typing import List, Tuple, Union, Iterable, Dict, Optional

import h5py
import numpy as np
from dask.distributed import Client
from distributed import Queue
from scipy.spatial.distance import pdist, squareform

from fridata.models.manage_dataset.compute_batches import ComputeBatches
from fridata.models.manage_dataset.index.handle_index import read_index, create_index
from fridata.models.manage_dataset.index.handle_indexes import HandleIndexes
from fridata.models.manage_dataset.utils import read_pdbs_from_h5, format_time
from fridata.utlis.logging import log_title

from fridata.utlis.logging import logger


def read_coordinates_from_h5(h5_file_path: str, protein_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Read coordinates from H5 file for specified protein IDs.
    
    Args:
        h5_file_path: Path to the H5 file containing coordinates
        protein_ids: List of protein IDs to read coordinates for
        
    Returns:
        Dictionary mapping protein_id to coordinates array
    """
    coordinates = {}
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            for protein_id in protein_ids:
                if protein_id in f:
                    # Read coordinates from the 'coords' dataset
                    coords_data = f[protein_id]['coords'][:]
                    coordinates[protein_id] = coords_data
                else:
                    logger.warning(f"Protein {protein_id} not found in {h5_file_path}")
    except Exception as e:
        logger.error(f"Error reading coordinates from {h5_file_path}: {e}")
    
    return coordinates


def __process_coordinates__(h5_files: List[str], protein_ids: List[str]) -> List[Tuple[str, np.ndarray]]:
    """
    Process coordinates from H5 files to generate distograms.
    
    Args:
        h5_files: List of H5 file paths containing coordinates
        protein_ids: List of protein IDs to process
        
    Returns:
        List of tuples (protein_id, distogram)
    """
    results = []
    
    # Read coordinates from all H5 files
    all_coordinates: Dict[str, np.ndarray] = {}
    for h5_file in h5_files:
        read_coords = read_coordinates_from_h5(h5_file, protein_ids)
        all_coordinates.update(read_coords)
    
    for protein_id in protein_ids:
        if protein_id not in all_coordinates:
            logger.warning(f"No coordinates found for {protein_id}")
            # Return a 1x1 array to indicate single/no coordinates
            results.append((protein_id, np.array([[0]])))
            continue
        
        # remove first column (residue index)
        coords = np.delete(
            all_coordinates[protein_id],
            0,
            axis=1
        )
        
        # Handle case where coordinates are empty or have only one point
        if len(coords) <= 1:
            logger.warning(f"Only one or no coordinates in {protein_id}.")
            results.append((protein_id, np.array([[1]])))
            continue
        
        # Calculate pairwise distances using scipy pdist
        try:
            # pdist returns condensed distance matrix, we need to convert to square form
            # NaN coordinates will result in NaN distances
            distances = pdist(coords, metric='euclidean')
            distances = distances.astype(np.float32)
            distogram = squareform(distances)
            
            results.append((protein_id, distogram))
            
        except Exception as e:
            logger.error(f"Error calculating distogram for {protein_id}: {e}")
            results.append((protein_id, np.array([[0]])))
    
    return results


def __save_result_batch_to_h5__(
    hf: h5py.File, results: Iterable[Tuple[str, np.ndarray]]
):
    distogram_pdbs_saved = []
    for pdb_path, distogram in results:

        if distogram.shape == (1, 1):
            logger.info(f"Only one CA in {pdb_path}.")
        else:
            protein_grp = hf.create_group(pdb_path)

            if distogram.shape[0] < 3:  # no compression for small dataset
                protein_grp.create_dataset("distogram", data=distogram)
            else:
                protein_grp.create_dataset(
                    "distogram", data=distogram, compression="lzf", shuffle=True
                )
            distogram_pdbs_saved.append(pdb_path)
    return distogram_pdbs_saved


def generate_distograms(structures_dataset: "StructuresDataset"):
    log_title("Generating distograms")

    start = time.time()

    protein_index = read_index(structures_dataset.dataset_index_file_path(), structures_dataset.config.data_path)

    handle_indexes: HandleIndexes = structures_dataset._handle_indexes

    search_index_result = handle_indexes.full_handle("distograms", protein_index, structures_dataset.overwrite)

    distogram_index = search_index_result.present

    client: Client = structures_dataset._client

    # Read coordinates index to get coordinate file paths
    coordinates_index = read_index(structures_dataset.coordinates_index_path(), structures_dataset.config.data_path)

    distograms_path_obj = Path(structures_dataset.config.data_path) / "distograms"
    if not distograms_path_obj.exists():
        distograms_path_obj.mkdir(exist_ok=True, parents=True)

    path = distograms_path_obj / structures_dataset.dataset_dir_name()
    path.mkdir(exist_ok=True, parents=True)

    partial_result_data_q = Queue(name="partial_result_data_q")

    def run(input_data, machine):
        i, input_data = input_data
        data_f = client.submit(__process_coordinates_batch__, *input_data, coordinates_index, workers=machine)
        return client.submit(collect_parallel, data_f, i, workers=machine)

    def collect(result):
        partial_result_data_q.put(result)

    def collect_parallel(result, i):
        if len(result) == 0:
            return {}
        distograms_file = path / f"batch_{i}.h5"
        hf = h5py.File(distograms_file, "w")
        partial = __save_result_batch_to_h5__(hf, result)
        hf.close()
        return {id_: str(distograms_file) for id_ in partial}

    compute_batches = ComputeBatches(
        structures_dataset._client,
        run,
        collect,
        f"distograms_{structures_dataset.db_type}",
    )
    
    # Group missing proteins for batch processing
    inputs = (
        (i, (protein_ids,)) for i, protein_ids in enumerate(search_index_result.grouped_missing_proteins.values())
    )

    compute_batches.compute(inputs, 5)

    distogram_pdbs_saved_futures = partial_result_data_q.get(batch=True)

    distogram_pdbs_saved = client.gather(distogram_pdbs_saved_futures)

    for dict_id_to_h5_file in distogram_pdbs_saved:
        if not hasattr(dict_id_to_h5_file, '__iter__'):
            logger.warning(f"Non-iterable result: {dict_id_to_h5_file}")
            continue
        distogram_index.update(dict_id_to_h5_file)

    ids = list(distogram_index.keys())
    for protein_id in ids:
        distogram_index[protein_id.removesuffix(".pdb")] = distogram_index.pop(protein_id)

    create_index(structures_dataset.distograms_index_path(), distogram_index, structures_dataset.config.data_path)

    end = time.time()
    logger.info(f"Total time: {format_time(end - start)}")


def __process_coordinates_batch__(protein_ids: List[str], coordinates_index: Dict[str, str]) -> List[Tuple[str, np.ndarray]]:
    """
    Process a batch of protein IDs to generate distograms from coordinates.
    
    Args:
        protein_ids: List of protein IDs to process
        coordinates_index: Index mapping protein IDs to coordinate file paths
        
    Returns:
        List of tuples (protein_id, distogram)
    """
    # Get unique H5 files for this batch
    h5_files = set()
    batch_protein_ids = []
    
    for protein_id in protein_ids:
        if protein_id in coordinates_index:
            h5_files.add(coordinates_index[protein_id])
            batch_protein_ids.append(protein_id)
        else:
            logger.warning(f"Protein {protein_id} not found in coordinates index")
    
    if not batch_protein_ids:
        return []
    
    return __process_coordinates__(list(h5_files), batch_protein_ids)


def read_distograms_from_file(
    distograms_file: Union[str, pathlib.Path], limit_keys: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Reads distograms from a h5py file.

    :param limit_keys:
    :param distograms_file: Path to the h5py file.
    :return: distograms: A dictionary mapping pdb_path to the corresponding distogram.
    """
    distograms = {}

    try:
        with h5py.File(distograms_file, "r") as hf:
            keys = hf.keys()
            if limit_keys is not None:
                keys = list(keys)[:limit_keys]
            for pdb_path in keys:
                distogram = hf[pdb_path]["distogram"][:]
                distograms[pdb_path] = distogram
    except IOError:
        logger.error(f"Error while reading the file {distograms_file}")
    return distograms
