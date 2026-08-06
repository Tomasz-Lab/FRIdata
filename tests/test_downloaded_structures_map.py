"""Unit tests for downloaded_structures.idx helper maps (no network)."""

from fridata.models.manage_dataset.utils import (
    canonical_afdb_uniprot_id,
    downloaded_structures_map_afdb_fetched,
    downloaded_structures_map_rcsb,
    extract_afdb_uniprot_id,
    is_afdb_downloadable_id,
    partition_afdb_download_ids,
)


def test_canonical_afdb_uniprot_id():
    assert canonical_afdb_uniprot_id("A0A009IHW8") == "A0A009IHW8"
    assert canonical_afdb_uniprot_id("AF-A0A009IHW8-F1-model_v4") == "A0A009IHW8"
    assert canonical_afdb_uniprot_id("AF-A0A009IHW8-F1-model_v6") == "A0A009IHW8"
    assert canonical_afdb_uniprot_id("A0A009IHW8-F1-model") == "A0A009IHW8"


def test_downloaded_structures_map_rcsb():
    agg = (["a_A"], ["content"], [("1abc", None), ("2def", "cifdata")])
    assert downloaded_structures_map_rcsb(agg, "/data/structures/pdb/0/pdbs.h5") == {
        "2def": "/data/structures/pdb/0/pdbs.h5",
    }
    assert downloaded_structures_map_rcsb(agg, "") == {}
    assert downloaded_structures_map_rcsb(agg, None) == {}


def test_extract_afdb_uniprot_id():
    assert extract_afdb_uniprot_id("A0A009IHW8") == "A0A009IHW8"
    assert extract_afdb_uniprot_id("AF-A0A009IHW8-F1-model_v4") == "A0A009IHW8"
    assert extract_afdb_uniprot_id("P12345") == "P12345"
    assert extract_afdb_uniprot_id("AF-URS00001F0C23_10116-F1-model_v4") is None
    assert extract_afdb_uniprot_id("AF-CPX-1033-F1-model_v4") is None


def test_partition_afdb_download_ids():
    ids = [
        "P12345",
        "AF-A0A009IHW8-F1-model_v4",
        "AF-URS00001F0C23_10116-F1-model_v4",
        "AF-CPX-1033-F1-model_v4",
    ]
    downloadable, skipped = partition_afdb_download_ids(ids)
    assert downloadable == ["P12345", "AF-A0A009IHW8-F1-model_v4"]
    assert skipped == [
        "AF-URS00001F0C23_10116-F1-model_v4",
        "AF-CPX-1033-F1-model_v4",
    ]
    assert is_afdb_downloadable_id("P12345")
    assert not is_afdb_downloadable_id("AF-CPX-1033-F1-model_v4")


def test_downloaded_structures_map_afdb_fetched():
    m = downloaded_structures_map_afdb_fetched(
        ["AF-U1-F1-model_v4", "U2"],
        "/data/structures/afdb/0/pdbs.h5",
    )
    assert m == {
        "U1": "/data/structures/afdb/0/pdbs.h5",
        "U2": "/data/structures/afdb/0/pdbs.h5",
    }
    assert downloaded_structures_map_afdb_fetched(["U1"], "") == {}
