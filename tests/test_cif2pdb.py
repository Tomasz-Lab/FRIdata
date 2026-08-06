import shutil

import pytest
import textwrap 

from fridata.models.manage_dataset.collection_type import CollectionType
from fridata.models.manage_dataset.database_type import DatabaseType
from fridata.models.manage_dataset.structures_dataset import StructuresDataset
from fridata.models.manage_dataset.utils import read_pdbs_from_h5
from fridata.models.utils.cif2pdb import cif_to_pdb
from fridata.config import Config
from tests.utils import compare_pdb_files, create_temp_txt_file
from pathlib import Path

# ===================================================
# Testing behaviour of the cif2pdb.cif2pdb() function
# ===================================================


# Based on https://github.com/PawelSzczerbiak/cif2pdb/blob/main/cif2pdb/tests/test_convert.py
# Different databases encode their PDB IDs in a different way
# especially, regarding chain and residue range (below).
# CATH:
# - chain: AUTHOR-defined
# - residue number: PDB-defined
# PDB90 etc.:
# - chain: AUTHOR-defined
# - residue number: all residues
# SCOP:
# - chain: AUTHOR-defined
# - residue number: AUTHOR-defined

CIF_HEADERS = textwrap.dedent("""
    loop_
    _atom_site.group_PDB
    _atom_site.id
    _atom_site.label_atom_id
    _atom_site.label_alt_id
    _atom_site.label_comp_id
    _atom_site.auth_asym_id
    _atom_site.label_seq_id
    _atom_site.pdbx_PDB_ins_code
    _atom_site.Cartn_x
    _atom_site.Cartn_y
    _atom_site.Cartn_z
    _atom_site.occupancy
    _atom_site.B_iso_or_equiv
    _atom_site.type_symbol
    _atom_site.pdbx_formal_charge
    _atom_site.pdbx_PDB_model_num
    """)

# Default input and output paths
INPATH = Path(__file__).parent / "data" / "cif"
OUTPATH = Path(__file__).parent / "data" / "pdb_generated"
EXPPATH = Path(__file__).parent / "data" / "pdb_expected"

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
    # # Cleanup after tests (optional)
    # for f in OUTPATH.glob('*'):
    #     if f.is_file():
    #         f.unlink()
    #     elif f.is_dir():
    #         shutil.rmtree(f)


@pytest.fixture
def setup_dataset():

    config = Config(
        data_path=str(OUTPATH),
        disto_type="CA",
        disto_thr="inf",
        separator="-",
        batch_size=1000,
    )

    dataset = StructuresDataset(
        db_type=DatabaseType.PDB,
        collection_type=CollectionType.subset,
        ids_file=create_temp_txt_file(['2fjh', ]),
        overwrite=True,
        config=config
    )
    dataset.create_dataset()
    yield dataset
    
    # Properly close the Dask client and cluster before cleanup
    if dataset._client is not None:
        try:
            # Close client first
            dataset._client.close()
            # Try to close the cluster if it exists
            if hasattr(dataset._client, 'cluster') and dataset._client.cluster is not None:
                dataset._client.cluster.close()
        except Exception:
            # Ignore any cleanup errors
            pass
        finally:
            dataset._client = None
    
    shutil.rmtree(dataset.dataset_repo_path())
    shutil.rmtree(dataset.dataset_path())


# BASIC TESTS

# TODO: should throw a warning (add logging firts)
@pytest.mark.skip(reason="Should print a warning")
def test_cif_to_pdb_none_atoms():
    """Test when no atoms are found"""
    # Empty CIF string should return None

    result = cif_to_pdb("", "1abc")
    
    assert result is None


def test_cif_to_pdb_basic():
    """Test basic CIF to PDB conversion"""
    # Then in test_cif_to_pdb_basic:
    cif_str = CIF_HEADERS + textwrap.dedent("""
        ATOM 1 N . ALA A 1 ? 1.000 2.000 3.000 1.00 20.00 N ? 1
        ATOM 2 CA . ALA A 1 ? 2.000 3.000 4.000 1.00 20.00 C ? 1
        """).lstrip()
    
    result = cif_to_pdb(cif_str, "1xyz")

    # Should return dict with chain A
    assert "1xyz_A" in result  

    # PDB content should contain formatted ATOM records
    pdb_lines = result["1xyz_A"].split("\n")
    nonempty_lines = sum(line.startswith("ATOM") for line in pdb_lines)
    assert 2 == nonempty_lines
    
    # Check if atom details are preserved
    assert "ALA" in pdb_lines[0]
    assert "A" in pdb_lines[0]
    assert "1.000" in pdb_lines[0]


def test_cif_to_pdb_multiple_chains():
    """Test conversion with multiple chains"""
    cif_str = CIF_HEADERS + textwrap.dedent("""
        ATOM 1 N . ALA A 1 ? 1.000 2.000 3.000 1.00 20.00 N ? 1
        ATOM 1 CA . ASP B 1 ? 4.000 5.000 6.000 1.00 20.00 N ? 1
        ATOM 2 N . ASP B 1 ? 4.000 3.000 6.000 1.00 10.00 N ? 1
        """).lstrip()
    
    result = cif_to_pdb(cif_str, "1xyz")

    # Result should have both chains
    assert "1xyz_A" in result
    assert "1xyz_B" in result

    # Chain A should have 1 atom
    pdb_lines = result["1xyz_A"].split("\n")
    nonempty_lines = sum(line.startswith("ATOM") for line in pdb_lines)
    assert 1 == nonempty_lines

    # Chain B should have 2 atoms
    pdb_lines = result["1xyz_B"].split("\n")
    nonempty_lines = sum(line.startswith("ATOM") for line in pdb_lines)
    assert 2 == nonempty_lines


def test_cif_to_pdb_invalid_residues():
    """Test handling of invalid residues"""
    cif_str = CIF_HEADERS + textwrap.dedent("""
        ATOM 1 N . ABC A 1 ? 1.000 2.000 3.000 1.00 20.00 N ? 1
        """).lstrip()
    
    result = cif_to_pdb(cif_str, "1xyz")
    
    # Should return empty dict since residue ABC is invalid
    assert result['1xyz_A'] == ''


# REAL DATA TESTS

def test_convert_pdb_2fjhL(setup_dataset):
    # Indirect testing of dataset creation
    # Indirect testing of cif2pdb conversion
    # AUTHORS chain different from PDB

    hdf5_files = [file for file in setup_dataset.structures_path().rglob('pdbs.h5')]
    assert len(hdf5_files) > 0
    pdbs = read_pdbs_from_h5(hdf5_files[0], None)
    
    # Conversion
    result = pdbs["2fjh_L"]  # Choose chain L
    
    with open(OUTPATH / '2fjh_L.pdb', 'w') as f:
        f.write(result)

    compare_pdb_files(OUTPATH / '2fjh_L.pdb',
                      EXPPATH / '2fjh_L.pdb')


def test_convert_pdb_5uyl32():
    # Multi-letter AUTHORS chain

    with open(INPATH / '5uyl.cif', 'r') as f:
        cif_str = f.read()
    
    # Conversion
    result = cif_to_pdb(cif_str, "5uyl")
    result = result["5uyl_32"]  # Choose chain 32
    
    with open(OUTPATH / '5uyl_32.pdb', 'w') as f:
        f.write(result)

    compare_pdb_files(OUTPATH / '5uyl_32.pdb',
                      EXPPATH / '5uyl_32.pdb')


def test_convert_pdb_1aa6A():
    # Non-standard aminoacid change (SEC into CYS) - residue 140
    # Occupancy different from 1.0 - residues 137-138

    with open(INPATH / '1aa6.cif', 'r') as f:
        cif_str = f.read()
    
    # Conversion
    result = cif_to_pdb(cif_str, "1aa6")
    result = result["1aa6_A"]  # Choose chain A
    
    with open(OUTPATH / '1aa6_A.pdb', 'w') as f:
        f.write(result)

    compare_pdb_files(OUTPATH / '1aa6_A.pdb',
                      EXPPATH / '1aa6_A.pdb')
