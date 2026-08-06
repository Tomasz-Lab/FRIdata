import h5py
import inspect
from pathlib import Path

import pytest

from fridata.models.manage_dataset.sequences.sequence_and_coordinates_retriever import (
    __get_sequences_and_coordinates_from_batch__,
)

INPATH = Path(__file__).parent / "data" / "distograms_pdbs"
OUTPATH = Path(__file__).parent / "data" / "selective_retrieval_generated"

OUTPATH.mkdir(parents=True, exist_ok=True)

PROTEIN = "2fjh_L"
PDB_CODE = f"{PROTEIN}.pdb"


@pytest.fixture
def batch_output_path(tmp_path):
    return tmp_path / inspect.currentframe().f_code.co_name


def test_batch_sequences_only(batch_output_path):
    sequence_lines, coords_mapping = __get_sequences_and_coordinates_from_batch__(
        INPATH / "pdbs.h5",
        [PDB_CODE],
        "CA",
        str(batch_output_path),
        is_sequences_retrieved=True,
        is_coordinates_retrieved=False,
    )

    assert len(sequence_lines) == 1
    assert sequence_lines[0].startswith(f">{PROTEIN}\n")
    assert coords_mapping == {}
    assert not Path(f"{batch_output_path}.h5").exists()


def test_batch_coordinates_only(batch_output_path):
    sequence_lines, coords_mapping = __get_sequences_and_coordinates_from_batch__(
        INPATH / "pdbs.h5",
        [PDB_CODE],
        "CA",
        str(batch_output_path),
        is_sequences_retrieved=False,
        is_coordinates_retrieved=True,
    )

    assert sequence_lines == []
    assert PROTEIN in coords_mapping
    h5_path = Path(coords_mapping[PROTEIN])
    assert h5_path.exists()

    with h5py.File(h5_path, "r") as f:
        assert PROTEIN in f
        assert "coords" in f[PROTEIN]


def test_batch_sequences_and_coordinates(batch_output_path):
    sequence_lines, coords_mapping = __get_sequences_and_coordinates_from_batch__(
        INPATH / "pdbs.h5",
        [PDB_CODE],
        "CA",
        str(batch_output_path),
        is_sequences_retrieved=True,
        is_coordinates_retrieved=True,
    )

    assert len(sequence_lines) == 1
    assert PROTEIN in coords_mapping
    assert Path(coords_mapping[PROTEIN]).exists()
