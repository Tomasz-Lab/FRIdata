import os
import shutil
import pytest
import time
from pathlib import Path
from dask.distributed import Client, LocalCluster
import logging
import warnings
import distributed
import tempfile

from tests.paths import OUTPATH
from toolbox.config import Config
from toolbox.models.manage_dataset.structures_dataset import StructuresDataset
from toolbox.models.manage_dataset.collection_type import CollectionType
from toolbox.models.manage_dataset.database_type import DatabaseType
from toolbox.models.manage_dataset.index.handle_index import read_index
from toolbox.models.manage_dataset.utils import canonical_afdb_uniprot_id

# give 666 permissions to new files
os.umask(0o002)

n_cores = os.cpu_count()

# Configure distributed logging to avoid conflicts
distributed_logger = logging.getLogger('distributed')
distributed_logger.setLevel(logging.WARNING)
if not distributed_logger.handlers:
    null_handler = logging.NullHandler()
    distributed_logger.addHandler(null_handler)

# Silence specific Dask warnings
warnings.simplefilter("ignore", distributed.comm.core.CommClosedError)
warnings.filterwarnings(
    "ignore",
    message=".*Creating scratch directories is taking a surprisingly long time.*",
)

# Create a Dask cluster for AFDB tests
afdb_cluster = LocalCluster(
    n_workers=n_cores - 1,
    threads_per_worker=1,
    memory_limit="4 GiB",
    silence_logs=logging.CRITICAL,
    worker_dashboard_address=None,
    dashboard_address="0.0.0.0:8991",  # Different port from test_dataset.py
)
afdb_client = Client(afdb_cluster)


@pytest.fixture(scope="module", autouse=True)
def clean_afdb_test_files(tmp_path_factory):
    """Clean up and prepare output directory for AFDB tests"""
    afdb_outpath = OUTPATH / "afdb_tests"
    afdb_outpath.mkdir(parents=True, exist_ok=True)
    
    # Clean existing files
    for f in afdb_outpath.glob('*'):
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)
    
    yield afdb_outpath
    
    # Cleanup after tests
    try:
        if afdb_client and afdb_client.status != "closed":
            afdb_client.close()
        if afdb_cluster:
            afdb_cluster.close()
    except Exception:
        pass


def create_afdb_dataset(ids_file_path: Path, dataset_name: str, config: Config) -> StructuresDataset:
    """
    Helper function to create an AFDB StructuresDataset instance.
    
    Args:
        ids_file_path: Path to the file containing UniProt IDs
        dataset_name: Name for the dataset version
        config: Config object with data paths and settings
    
    Returns:
        StructuresDataset instance with client attached
    """
    dataset = StructuresDataset(
        db_type=DatabaseType.AFDB,
        collection_type=CollectionType.subset,
        version=dataset_name,
        ids_file=ids_file_path,
        overwrite=True,
        config=config,
    )
    dataset._client = afdb_client
    return dataset


def create_temp_ids_file(ids: list) -> Path:
    """
    Create a temporary file with the given IDs.
    
    Args:
        ids: List of ID strings
    
    Returns:
        Path to the temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    temp_file.write('\n'.join(ids))
    temp_file.close()
    return Path(temp_file.name)


def test_afdb_download_full_format():
    """Test AFDB download with full format IDs (AF-A0A009IHW8-F1-model_v4)"""
    
    # Test IDs in full format
    test_ids = [
        "AF-A0A009IHW8-F1-model_v4",
        "AF-A0A023FDY8-F1-model_v4",
        "AF-A0A023FFD0-F1-model_v4",
        "AF-A0A023I7E1-F1-model_v4",
    ]
    
    dataset_name = "afdb_full_format"
    afdb_outpath = OUTPATH / "afdb_tests"
    
    config = Config(
        data_path=str(afdb_outpath),
        disto_type="CA",
        disto_thr="inf",
        separator="-",
        batch_size=1000,
    )
    
    # Create temporary IDs file
    ids_file = create_temp_ids_file(test_ids)
    
    try:
        # Create dataset and download
        dataset = create_afdb_dataset(ids_file, dataset_name, config)
        
        print(f"\n=== Starting AFDB download test (full format) ===")
        start_time = time.time()
        dataset.create_dataset()
        download_time = time.time() - start_time
        print(f"Download took {download_time:.2f} seconds")
        
        # Verify dataset directory structure exists
        dataset_path = dataset.dataset_path()
        assert dataset_path.exists(), f"Dataset path does not exist: {dataset_path}"
        
        # Verify dataset index file exists
        index_file = dataset.dataset_index_file_path()
        assert index_file.exists(), f"Dataset index file does not exist: {index_file}"
        
        # Read and verify index
        index = read_index(index_file, config.data_path)
        assert len(index) > 0, "Dataset index is empty"
        
        # Verify that all expected UniProt IDs are present (may have chain suffixes)
        # IDs should be processed to remove AF- prefix and -F1-model_v4 suffix
        expected_uniprot_ids = ["A0A009IHW8", "A0A023FDY8", "A0A023FFD0", "A0A023I7E1"]
        
        # Check that at least one entry exists for each UniProt ID
        # Index keys will be in format like "A0A009IHW8_A" (with chain suffix)
        found_ids = {
            canonical_afdb_uniprot_id(key.split("_", 1)[0]) for key in index.keys()
        }

        for uniprot_id in expected_uniprot_ids:
            assert uniprot_id in found_ids, f"UniProt ID {uniprot_id} not found in dataset index"
        
        # Verify HDF5 files exist
        structures_path = dataset.structures_path()
        h5_files = list(structures_path.glob("**/*.h5"))
        assert len(h5_files) > 0, "No HDF5 files found in structures directory"

        dl_idx_path = dataset.downloaded_structures_index_path()
        assert dl_idx_path.exists(), f"downloaded_structures.idx missing: {dl_idx_path}"
        dl_index = read_index(dl_idx_path, config.data_path)
        assert len(dl_index) > 0, "downloaded_structures.idx is empty"
        for uniprot_id in expected_uniprot_ids:
            assert uniprot_id in dl_index, f"UniProt {uniprot_id} not in downloaded_structures.idx"
            h5_path = Path(dl_index[uniprot_id])
            assert h5_path.is_file(), f"pdbs.h5 path missing for {uniprot_id}: {h5_path}"
            assert h5_path.name == "pdbs.h5", f"expected pdbs.h5, got {h5_path.name}"
        
        print(f"=== Completed AFDB download test (full format) ===\n")
        
    finally:
        # Cleanup temporary file
        if ids_file.exists():
            ids_file.unlink()


def test_afdb_download_stripped_format():
    """Test AFDB download with stripped UniProt IDs (A0A009IHW8)"""
    
    # Test IDs in stripped format
    test_ids = [
        "A0A009IHW8",
        "A0A023FDY8",
        "A0A023FFD0",
        "A0A023I7E1",
    ]
    
    dataset_name = "afdb_stripped_format"
    afdb_outpath = OUTPATH / "afdb_tests"
    
    config = Config(
        data_path=str(afdb_outpath),
        disto_type="CA",
        disto_thr="inf",
        separator="-",
        batch_size=1000,
    )
    
    # Create temporary IDs file
    ids_file = create_temp_ids_file(test_ids)
    
    try:
        # Create dataset and download
        dataset = create_afdb_dataset(ids_file, dataset_name, config)
        
        print(f"\n=== Starting AFDB download test (stripped format) ===")
        start_time = time.time()
        dataset.create_dataset()
        download_time = time.time() - start_time
        print(f"Download took {download_time:.2f} seconds")
        
        # Verify dataset directory structure exists
        dataset_path = dataset.dataset_path()
        assert dataset_path.exists(), f"Dataset path does not exist: {dataset_path}"
        
        # Verify dataset index file exists
        index_file = dataset.dataset_index_file_path()
        assert index_file.exists(), f"Dataset index file does not exist: {index_file}"
        
        # Read and verify index
        index = read_index(index_file, config.data_path)
        assert len(index) > 0, "Dataset index is empty"
        
        # Verify that all expected UniProt IDs are present
        expected_uniprot_ids = ["A0A009IHW8", "A0A023FDY8", "A0A023FFD0", "A0A023I7E1"]
        
        found_ids = {
            canonical_afdb_uniprot_id(key.split("_", 1)[0]) for key in index.keys()
        }

        for uniprot_id in expected_uniprot_ids:
            assert uniprot_id in found_ids, f"UniProt ID {uniprot_id} not found in dataset index"
        
        # Verify HDF5 files exist
        structures_path = dataset.structures_path()
        h5_files = list(structures_path.glob("**/*.h5"))
        assert len(h5_files) > 0, "No HDF5 files found in structures directory"

        dl_idx_path = dataset.downloaded_structures_index_path()
        assert dl_idx_path.exists(), f"downloaded_structures.idx missing: {dl_idx_path}"
        dl_index = read_index(dl_idx_path, config.data_path)
        for uniprot_id in expected_uniprot_ids:
            assert uniprot_id in dl_index, f"UniProt {uniprot_id} not in downloaded_structures.idx"
        
        print(f"=== Completed AFDB download test (stripped format) ===\n")
        
    finally:
        # Cleanup temporary file
        if ids_file.exists():
            ids_file.unlink()


def test_afdb_download_mixed_format():
    """Test AFDB download with mixed ID formats"""
    
    # Test IDs in mixed format (both full and stripped)
    test_ids = [
        "AF-A0A009IHW8-F1-model_v4",
        "A0A023FDY8",
        "AF-A0A023FFD0-F1-model_v4",
        "A0A023I7E1",
    ]
    
    dataset_name = "afdb_mixed_format"
    afdb_outpath = OUTPATH / "afdb_tests"
    
    config = Config(
        data_path=str(afdb_outpath),
        disto_type="CA",
        disto_thr="inf",
        separator="-",
        batch_size=1000,
    )
    
    # Create temporary IDs file
    ids_file = create_temp_ids_file(test_ids)
    
    try:
        # Create dataset and download
        dataset = create_afdb_dataset(ids_file, dataset_name, config)
        
        print(f"\n=== Starting AFDB download test (mixed format) ===")
        start_time = time.time()
        dataset.create_dataset()
        download_time = time.time() - start_time
        print(f"Download took {download_time:.2f} seconds")
        
        # Verify dataset directory structure exists
        dataset_path = dataset.dataset_path()
        assert dataset_path.exists(), f"Dataset path does not exist: {dataset_path}"
        
        # Verify dataset index file exists
        index_file = dataset.dataset_index_file_path()
        assert index_file.exists(), f"Dataset index file does not exist: {index_file}"
        
        # Read and verify index
        index = read_index(index_file, config.data_path)
        assert len(index) > 0, "Dataset index is empty"
        
        # Verify that all expected UniProt IDs are present
        # Regardless of input format, all should be processed correctly
        expected_uniprot_ids = ["A0A009IHW8", "A0A023FDY8", "A0A023FFD0", "A0A023I7E1"]
        
        found_ids = {
            canonical_afdb_uniprot_id(key.split("_", 1)[0]) for key in index.keys()
        }

        for uniprot_id in expected_uniprot_ids:
            assert uniprot_id in found_ids, f"UniProt ID {uniprot_id} not found in dataset index"
        
        # Verify HDF5 files exist
        structures_path = dataset.structures_path()
        h5_files = list(structures_path.glob("**/*.h5"))
        assert len(h5_files) > 0, "No HDF5 files found in structures directory"

        dl_idx_path = dataset.downloaded_structures_index_path()
        assert dl_idx_path.exists(), f"downloaded_structures.idx missing: {dl_idx_path}"
        dl_index = read_index(dl_idx_path, config.data_path)
        for uniprot_id in expected_uniprot_ids:
            assert uniprot_id in dl_index, f"UniProt {uniprot_id} not in downloaded_structures.idx"
        
        print(f"=== Completed AFDB download test (mixed format) ===\n")
        
    finally:
        # Cleanup temporary file
        if ids_file.exists():
            ids_file.unlink()


def test_afdb_id_conversion():
    """Test that ID conversion works correctly in _download_afdb_"""
    
    # Test that IDs are correctly converted from full format to stripped format
    test_ids_full = [
        "AF-A0A009IHW8-F1-model_v4",
        "AF-A0A023FDY8-F1-model_v4",
    ]
    
    test_ids_stripped = [
        "A0A009IHW8",
        "A0A023FDY8",
    ]
    
    afdb_outpath = OUTPATH / "afdb_tests"
    
    config = Config(
        data_path=str(afdb_outpath),
        disto_type="CA",
        disto_thr="inf",
        separator="-",
        batch_size=1000,
    )
    
    # Test with full format
    ids_file_full = create_temp_ids_file(test_ids_full)
    dataset_full = create_afdb_dataset(ids_file_full, "afdb_conversion_full", config)
    
    # Test with stripped format
    ids_file_stripped = create_temp_ids_file(test_ids_stripped)
    dataset_stripped = create_afdb_dataset(ids_file_stripped, "afdb_conversion_stripped", config)
    
    try:
        # Create both datasets
        dataset_full.create_dataset()
        dataset_stripped.create_dataset()
        
        # Read both indices
        index_full = read_index(dataset_full.dataset_index_file_path(), config.data_path)
        index_stripped = read_index(dataset_stripped.dataset_index_file_path(), config.data_path)
        
        # Extract base UniProt IDs from both indices
        ids_full = {
            canonical_afdb_uniprot_id(key.split("_", 1)[0]) for key in index_full.keys()
        }
        ids_stripped = {
            canonical_afdb_uniprot_id(key.split("_", 1)[0]) for key in index_stripped.keys()
        }
        
        # Both should result in the same set of UniProt IDs
        assert ids_full == ids_stripped, \
            f"ID conversion failed: full format produced {ids_full}, stripped format produced {ids_stripped}"
        
    finally:
        # Cleanup temporary files
        if ids_file_full.exists():
            ids_file_full.unlink()
        if ids_file_stripped.exists():
            ids_file_stripped.unlink()


