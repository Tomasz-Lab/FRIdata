from pathlib import Path

import pytest

from tests.paths import EXPPATH

from toolbox.viewer.export_index_html import (
    discover_dataset,
    extract_dataset_identity_from_path,
    extract_batch_id_from_path,
    stream_parse_idx,
    compute_index_stats,
    compute_global_rollup,
    find_index_files,
)


def test_discover_dataset_single_ref():
    root = EXPPATH / "datasets"
    ds_dir = root / "PDB-subset--fifth_7"

    by_path = discover_dataset(str(ds_dir), root)
    assert by_path.path == ds_dir.resolve()
    assert by_path.identity.slug == "fifth_7"

    by_slug = discover_dataset("fifth_7", root)
    assert by_slug.path.resolve() == ds_dir.resolve()

    by_folder_name = discover_dataset("PDB-subset--fifth_7", root)
    assert by_folder_name.path.resolve() == ds_dir.resolve()

    by_json = discover_dataset(str(ds_dir / "dataset.json"), root)
    assert by_json.path.resolve() == ds_dir.resolve()

    with pytest.raises(ValueError):
        discover_dataset(None, root)
    with pytest.raises(FileNotFoundError):
        discover_dataset("__nonexistent_dataset_slug__", root)


def test_identity_and_batch_parsing():
    # Pattern A: hyphenated dataset slug in path
    coord_path = "coordinates/PDB-subset--fourth_7/batch_0_ca.h5"
    id_a = extract_dataset_identity_from_path(coord_path)
    assert id_a is not None
    assert id_a.db == "PDB"
    assert id_a.collection == "subset"
    assert id_a.slug == "fourth_7"
    assert extract_batch_id_from_path(coord_path, "coordinates") == "0"

    # Pattern B: structured path under structures/<DB>/<COLL>_<slug>/<batch>/pdbs.h5
    ds_path = "structures/PDB/subset_/fourth_7/0/pdbs.h5"
    id_b = extract_dataset_identity_from_path(ds_path)
    assert id_b is not None
    assert id_b.db == "PDB"
    assert id_b.collection == "subset"
    assert id_b.slug == "fourth_7"
    assert extract_batch_id_from_path(ds_path, "dataset") == "0"


def test_stream_and_stats_coordinates():
    dataset_dir = EXPPATH / "datasets" / "PDB-subset--fifth_7"
    fwd = dataset_dir / "coordinates.idx"
    rev = dataset_dir / "coordinates_reversed.idx"

    # stream forward as (protein_id -> file_path)
    fwd_items = list(stream_parse_idx(fwd))
    assert len(fwd_items) > 0
    # all values should point to coordinates/.../batch_0_ca.h5
    unique_targets = {v for _, v in fwd_items}
    assert unique_targets == {"coordinates/PDB-subset--fourth_7/batch_0_ca.h5"}

    # stream reversed as (file_path -> [protein_ids])
    rev_items = list(stream_parse_idx(rev))
    assert len(rev_items) == 1
    rev_file, rev_list = rev_items[0]
    assert rev_file.endswith("batch_0_ca.h5")
    assert isinstance(rev_list, list) and len(rev_list) == len(fwd_items)

    stats = compute_index_stats("coordinates", fwd, rev)
    assert stats["forward_present"] is True
    assert stats["reversed_present"] is True
    assert stats["num_proteins_forward"] == len(fwd_items)
    assert stats["num_files_referenced"] == 1
    assert stats["num_edges_reversed"] == len(fwd_items)

    # by_dataset keyed by referenced slug (fourth_7)
    by_ds = stats["by_dataset"]
    assert "fourth_7" in by_ds
    assert by_ds["fourth_7"]["files_referenced"] == 1
    assert by_ds["fourth_7"]["proteins_referencing"] == len(fwd_items)
    # batch table present with batch id '0'
    assert by_ds["fourth_7"]["files_per_batch"].get("0") == 1


def test_global_rollup_with_multiple_index_types():
    dataset_dir = EXPPATH / "datasets" / "PDB-subset--fifth_7"
    coord_fwd = dataset_dir / "coordinates.idx"
    coord_rev = dataset_dir / "coordinates_reversed.idx"
    ds_rev = dataset_dir / "dataset_reversed.idx"

    per_index = {}
    per_index["coordinates"] = compute_index_stats("coordinates", coord_fwd, coord_rev)
    per_index["dataset"] = compute_index_stats("dataset", None, ds_rev)

    global_roll = compute_global_rollup(per_index)
    agg = global_roll["aggregate"]
    assert "fourth_7" in agg
    # files_referenced should sum: 1 (coordinates) + 1 (dataset) = 2
    assert agg["fourth_7"]["files_referenced"] == 2
    # proteins_referencing should sum list lengths from both reversed indexes
    num_coord = per_index["coordinates"]["by_dataset"]["fourth_7"]["proteins_referencing"]
    num_ds = per_index["dataset"]["by_dataset"]["fourth_7"]["proteins_referencing"]
    assert agg["fourth_7"]["proteins_referencing"] == (num_coord + num_ds)


def test_find_index_files_presence():
    dataset_dir = EXPPATH / "datasets" / "PDB-subset--fifth_7"
    found = find_index_files(dataset_dir)
    # Coordinates and dataset reversed should be present in fixtures
    assert found["coordinates"].reversed is not None
    assert found["dataset"].reversed is not None
    # Forward coordinates present
    assert found["coordinates"].forward is not None

