import os
import shutil
import json
import pytest
import time
from unittest.mock import patch
from dask.distributed import Client, LocalCluster
import logging
import warnings
import distributed

from fridata.config import Config, CarbonAtomType
from tests.paths import EXPPATH
from fridata.models.embedding.embedder.embedder_type import EmbedderType
from fridata.models.manage_dataset.index.handle_indexes import IndexableAbstraction

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

cluster = LocalCluster(
    n_workers=n_cores - 1,  # Use fewer workers for testing
    threads_per_worker=1,
    memory_limit="4 GiB",
    silence_logs=logging.CRITICAL,
    worker_dashboard_address=None,
    dashboard_address="0.0.0.0:8990",  # Use different port to avoid conflicts
)
client = Client(cluster)

@pytest.fixture(scope="session", autouse=True)
def cleanup_global_dask():
    """Clean up the global Dask client and cluster"""
    global client, cluster
    try:
        if client and client.status != "closed":
            client.close()
        if cluster:
            cluster.close()
    except Exception:
        pass
    finally:
        client = None
        cluster = None


def test_embeddings_for_esm2_ca_present():
    found_items = perform_index_search("embeddings", EmbedderType.ESM2_T30_150M, "CA")
    assert len(found_items) > 0

def test_embeddings_for_esmc_ca_missing():
    found_items = perform_index_search("embeddings", EmbedderType.ESMC_600M, "CA")

    assert len(found_items) == 0

def test_distograms_for_ca_present():
    found_items = perform_index_search("distograms", EmbedderType.ESMC_600M, "CA")

    assert len(found_items) > 0

def test_distograms_for_cb_missing():
    found_items = perform_index_search("distograms", EmbedderType.ESMC_600M, "CB")

    assert len(found_items) == 0

def perform_index_search(index_type: IndexableAbstraction, embedder_type: EmbedderType, disto_type: CarbonAtomType):
    from fridata.models.manage_dataset.index.handle_indexes import HandleIndexes

    dataset = create_dataset("test_dataset", embedder_type, disto_type)

    handle_indexes = HandleIndexes(dataset)

    handle_indexes.read_indexes(index_type)

    found_items = handle_indexes.file_paths_storage[index_type]

    return found_items

def create_dataset(
    dataset_name: str,
    embedder_type: EmbedderType,
    disto_type: CarbonAtomType,
    overwrite=False
):
    from fridata.models.manage_dataset.structures_dataset import StructuresDataset
    from fridata.models.manage_dataset.collection_type import CollectionType
    from fridata.models.manage_dataset.database_type import DatabaseType

    config = Config(
        data_path=str(EXPPATH),
        disto_type=disto_type,
        disto_thr="inf",
        separator="-",
        batch_size=1000,
    )

    dataset = StructuresDataset(
        db_type=DatabaseType.PDB,
        collection_type=CollectionType.subset,
        version=dataset_name,
        ids_file=EXPPATH / "input_lists" / f"pdb_5.txt",
        overwrite=overwrite,
        config=config,
        embedder_type=embedder_type,
    )
    dataset._client = client

    return dataset