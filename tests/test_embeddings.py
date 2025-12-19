import os
import glob
import pytest
import textwrap 
import numpy as np
import shutil

import h5py

from os.path import join

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
# TODO: @Adam: add test for ESM-C
# TODO: @Pawel: add expected output for ESM-C
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

# TODO
# Add tests for other embeddings (Ankh etc.)