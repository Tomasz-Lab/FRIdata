import os
import glob
import pytest
import textwrap 
import numpy as np
import shutil

import h5py

from os.path import join

# Embedding tests need the optional 'embeddings' extra (torch/esm/transformers).
# Skip the whole module gracefully when it is not installed so a core-only
# `pip install -e .[test]` checkout can still run the rest of the suite.
pytest.importorskip("torch", reason="install FRIdata[embeddings] to run embedding tests")

from toolbox.models.embedding.embedder.esm2_embedder import ESM2Embedder
from tests.utils import compare_pdb_files
from pathlib import Path


# =======================================================
# Testing behaviour of the ESM2Embedder class
# =======================================================

# Default paths
EXPPATH = Path(__file__).parent / "data" / "embeddings_expected"
OUTPATH = Path(__file__).parent / "data" / "embeddings_generated"

# Global variables
SEQ_1 = "MKVLLYIAASCLMLLALNVSAENTQQEEEDYDYG"
SEQ_2 = "_-XSVAAAVAGLLFGLDIGVIAGALPFITDHFVLTSRLQEW"
ESM_MODEL = "esm2_t33_650M_UR50D"
ESMC_MODEL = "esmc_300m"

@pytest.fixture(scope="session", autouse=True)
def clean_generated_files(tmp_path_factory):
    # Create output directory if it doesn't exist
    OUTPATH.mkdir(parents=True, exist_ok=True)
    # Clean existing files
    for f in OUTPATH.glob('*'):
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)
    # Verify directory is empty
    assert not list(OUTPATH.iterdir())
    yield
    # Cleanup after tests (optional)
    # for f in OUTPATH.glob('*'):
    #     f.unlink()


# NOTE: Embeddings generated with different environments
# may differ slightly and tests may fail due to this.
def test_esm():

    exp_1 = np.load(EXPPATH / f'{ESM_MODEL}__SEQ_1.npy') 
    exp_2 = np.load(EXPPATH / f'{ESM_MODEL}__SEQ_2.npy') 
    dict_ = {"SEQ_1": SEQ_1, "SEQ_2": SEQ_2}

    # Create ESM2Embedder instance and compute embeddings
    embedder = ESM2Embedder()
    embedder.embed(dict_, OUTPATH)

    # Load results - both sequences should be in batch_0.h5
    with h5py.File(OUTPATH / 'batch_0.h5', 'r') as f:
        res_1 = f['SEQ_1'][:]
        res_2 = f['SEQ_2'][:]

    assert np.allclose(exp_1, res_1, rtol=1e-4, atol=1e-5)
    assert np.allclose(exp_2, res_2, rtol=1e-4, atol=1e-5)

def test_esmc():
    # Expected data was generated with python 3.12 by the ESMCEmbedder itself,
    # which puts the model in eval() mode and runs under torch.inference_mode().
    # Keep both of these when you regenerate the data, or the values change.
    exp_1 = np.load(EXPPATH / f'{ESMC_MODEL}__SEQ_1.npy')
    exp_2 = np.load(EXPPATH / f'{ESMC_MODEL}__SEQ_2.npy')
    dict_ = {"SEQ_1": SEQ_1, "SEQ_2": SEQ_2}

    from toolbox.models.embedding.embedder.esmc_embedder import ESMCEmbedder

    # Own output directory so the ESM2 batch files are not overwritten.
    out_path = OUTPATH / ESMC_MODEL
    out_path.mkdir(parents=True, exist_ok=True)

    embedder = ESMCEmbedder(model_name=ESMC_MODEL)
    embedder.embed(dict_, out_path)

    with h5py.File(out_path / 'batch_0.h5', 'r') as f:
        res_1 = f['SEQ_1'][:]
        res_2 = f['SEQ_2'][:]

    assert res_1.shape == exp_1.shape
    assert res_2.shape == exp_2.shape
    assert np.allclose(exp_1, res_1, rtol=1e-4, atol=1e-5)
    assert np.allclose(exp_2, res_2, rtol=1e-4, atol=1e-5)


# TODO
# Add tests for other embeddings (Ankh etc.)


def test_embedding_run_persists_embedder_metadata(tmp_path):
    import json
    from unittest.mock import patch

    from toolbox.config import Config
    from toolbox.models.embedding.embedder.embedder_type import EmbedderType
    from toolbox.models.manage_dataset.collection_type import CollectionType
    from toolbox.models.manage_dataset.database_type import DatabaseType
    from toolbox.models.manage_dataset.structures_dataset import StructuresDataset

    config = Config(
        data_path=str(tmp_path),
        disto_type="CA",
        disto_thr="inf",
        separator="-",
        batch_size=1000,
    )
    dataset = StructuresDataset(
        db_type=DatabaseType.AFDB,
        collection_type=CollectionType.subset,
        version="embedder_metadata_test",
        config=config,
        embedder_type=None,
        embedding_size=None,
    )
    dataset_path = dataset.dataset_path()
    dataset_path.mkdir(parents=True)
    dataset.save_dataset_metadata()

    with patch(
        "toolbox.models.embedding.embedding.search_embedding_indexes"
    ) as mock_search, patch(
        "toolbox.models.embedding.embedding.Embedding.missing_ids_to_fasta",
        return_value={},
    ), patch(
        "toolbox.models.embedding.embedder.embedder_type.EmbedderType.create_embedder"
    ) as mock_create_embedder, patch(
        "toolbox.models.manage_dataset.index.handle_index.create_index"
    ):
        from toolbox.models.manage_dataset.index.handle_indexes import SearchIndexResult

        mock_search.return_value = SearchIndexResult(
            missing_protein_files={},
            present={},
            grouped_missing_proteins={},
        )
        mock_create_embedder.return_value.embed.return_value = {}

        dataset.embedder_type = EmbedderType.ESMC_600M
        dataset.generate_embeddings()

    saved = json.loads((dataset_path / "dataset.json").read_text())
    assert saved["embedder_type"] == "esmc_600m"
    assert saved["embedding_size"] == 1152