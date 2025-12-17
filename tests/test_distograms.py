import numpy as np
import inspect

from toolbox.models.manage_dataset.distograms.generate_distograms import __process_coordinates__
from toolbox.models.manage_dataset.sequences.sequence_and_coordinates_retriever import __get_sequences_and_coordinates_from_batch__
from pathlib import Path

# ==========================================================================================
# Testing behaviour of the distograms.generate_distograms.__process_coordinates__() function
# ==========================================================================================

# Default paths
INPATH = Path(__file__).parent / "data" / "distograms_pdbs"
OUTPATH = Path(__file__).parent / "data" / "distograms_generated"
EXPPATH = Path(__file__).parent / "data" / "distograms_expected"

RTOL = 1e-3  # TODO: small precision
ATOL = 1e-8

def test_full():
    # All residues present in the PDB

    name = '2fjh_L'
    name_out = inspect.currentframe().f_code.co_name

    __get_sequences_and_coordinates_from_batch__(INPATH / f"pdbs.h5", [f"{name}.pdb"], "CA", OUTPATH / name_out)
    coordinates = __process_coordinates__([OUTPATH / f"{name_out}.h5"], [name])
    
    expected = np.load(EXPPATH / f"{name}.npy")

    assert np.allclose(expected, coordinates[0][1], rtol=RTOL, atol=ATOL)


def test_cropped():
    # Missing residudes at the beginning (1-4)
    # Should be skipped in the distogram generation

    name = '1aa6_Acropped'
    name_out = inspect.currentframe().f_code.co_name

    __get_sequences_and_coordinates_from_batch__(INPATH / f"pdbs.h5", [f"{name}.pdb"], "CA", OUTPATH / name_out)
    coordinates = __process_coordinates__([OUTPATH / f"{name_out}.h5"], [name])
    expected = ~np.isnan(np.load(EXPPATH / f"{name}.npy"))

    assert np.allclose(expected, ~np.isnan(coordinates[0][1]), rtol=RTOL, atol=ATOL)


def test_missing_CA():
    # Missing CA atom in residue (1)  -- should be ignored since we have aminoacid name
    # Missing CA atom in residue (3)
    # Missing CB atom in residue (4)  -- should be ignored
    # Distogram type should be CA

    name = '1aa6_Amisssing_CA_or_CB'
    name_out = inspect.currentframe().f_code.co_name

    __get_sequences_and_coordinates_from_batch__(INPATH / f"pdbs.h5", [f"{name}.pdb"], "CA", OUTPATH / name_out)
    coordinates = __process_coordinates__([OUTPATH / f"{name_out}.h5"], [name])
    expected = np.load(EXPPATH / f"{name}_version_CA.npy")

    assert np.allclose(expected, coordinates[0][1], rtol=RTOL, atol=ATOL, equal_nan=True)


def test_missing_CB():
    # Missing CA atom in residue (1)  -- should be ignored
    # Missing CA atom in residue (3)  -- should be ignored
    # Missing CB atom in residues [34, 4, 39, 14, 24]
    # Distogram type should be CB

    name = '1aa6_Amisssing_CA_or_CB'
    name_out = inspect.currentframe().f_code.co_name

    __get_sequences_and_coordinates_from_batch__(INPATH / f"pdbs.h5", [f"{name}.pdb"], "CB", OUTPATH / name_out)
    coordinates = __process_coordinates__([OUTPATH / f"{name_out}.h5"], [name])
    expected = np.load(EXPPATH / f"{name}_version_CB.npy")

    assert np.allclose(expected, coordinates[0][1], rtol=RTOL, atol=ATOL, equal_nan=True)


def test_missing_one():
    # Missing residue in the middle (7)

    name = '5uyl_32oneX'
    name_out = inspect.currentframe().f_code.co_name

    __get_sequences_and_coordinates_from_batch__(INPATH / f"pdbs.h5", [f"{name}.pdb"], "CA", OUTPATH / name_out)
    coordinates = __process_coordinates__([OUTPATH / f"{name_out}.h5"], [name])
    expected = np.load(EXPPATH / f"{name}.npy")

    assert np.allclose(expected, coordinates[0][1], rtol=RTOL, atol=ATOL, equal_nan=True)


def test_missing_two():
    # Missing residues in the middle in two places (8-10, 41)

    name = '5uyl_32twoX'
    name_out = inspect.currentframe().f_code.co_name

    __get_sequences_and_coordinates_from_batch__(INPATH / f"pdbs.h5", [f"{name}.pdb"], "CA", OUTPATH / name_out)
    coordinates = __process_coordinates__([OUTPATH / f"{name_out}.h5"], [name])
    expected = np.load(EXPPATH / f"{name}.npy")

    assert np.allclose(expected, coordinates[0][1], rtol=RTOL, atol=ATOL, equal_nan=True)

