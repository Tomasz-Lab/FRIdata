from toolbox.scripts.archive import _shard_zip_name


def test_shard_zip_name_distinct_batches_same_basename():
    data = "/data"
    a = f"{data}/structures/PDB/subset_/fourth_7/0/pdbs.h5"
    b = f"{data}/structures/PDB/subset_/fourth_7/1/pdbs.h5"
    assert _shard_zip_name(a, data) != _shard_zip_name(b, data)
    assert _shard_zip_name(a, data).endswith(".zip")
    assert "0" in _shard_zip_name(a, data) or "pdbs" in _shard_zip_name(a, data)


def test_shard_zip_name_distinct_trees_same_batch_id():
    data = "/data"
    a = f"{data}/structures/PDB/subset_/fourth_7/0/pdbs.h5"
    b = f"{data}/structures/PDB/subset_/other_slug/0/pdbs.h5"
    assert _shard_zip_name(a, data) != _shard_zip_name(b, data)


def test_shard_zip_name_long_path_uses_hash():
    data = "/data"
    long_mid = "x" * 300
    h5 = f"{data}/structures/{long_mid}/0/pdbs.h5"
    name = _shard_zip_name(h5, data)
    assert name.endswith(".zip")
    assert len(name) < 80
