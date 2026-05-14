import zipfile
from pathlib import Path

import pytest

from toolbox.models.manage_dataset.utils import compress_and_save_h5
from toolbox.scripts.archive import (
    _shard_zip_name,
    merge_shard_zips_flat,
    normalize_pdb_codes_for_h5,
    process_h5_file,
)


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


def _write_pdb_zip(path: Path, members: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, body in members.items():
            zf.writestr(
                name,
                body,
                compress_type=zipfile.ZIP_DEFLATED,
            )


def test_merge_shard_zips_flat_combines_members(tmp_path):
    a = tmp_path / "a.zip"
    b = tmp_path / "b.zip"
    out = tmp_path / "merged.zip"
    _write_pdb_zip(a, {"1aaa_A.pdb": "HEADER 1\n"})
    _write_pdb_zip(b, {"2bbb_B.pdb": "HEADER 2\n"})
    merge_shard_zips_flat([b, a], out)
    with zipfile.ZipFile(out, "r") as zf:
        names = sorted(zf.namelist())
    assert names == ["1aaa_A.pdb", "2bbb_B.pdb"]


def test_normalize_pdb_codes_for_h5_expands_stem_and_pdb_suffix():
    assert normalize_pdb_codes_for_h5(["A0A1_A", "B2.pdb", "  ", "A0A1_A"]) == [
        "A0A1_A",
        "A0A1_A.pdb",
        "B2",
        "B2.pdb",
    ]


def test_process_h5_file_subset_writes_only_requested_pdbs(tmp_path):
    batch = tmp_path / "batch0"
    batch.mkdir()
    compress_and_save_h5(
        batch,
        (
            ["1aaa_A.pdb", "2bbb_B.pdb", "3ccc_C.pdb"],
            ["HEADER 1\n", "HEADER 2\n", "HEADER 3\n"],
            [],
        ),
    )
    h5 = batch / "pdbs.h5"
    out_dir = tmp_path / "staging"
    out_dir.mkdir()
    zip_path = process_h5_file(
        str(h5),
        str(tmp_path),
        out_dir,
        pdb_codes=["1aaa_A", "3ccc_C.pdb"],
    )
    assert zip_path is not None
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(zf.namelist())
        assert names == ["1aaa_A.pdb", "3ccc_C.pdb"]
        assert zf.read("1aaa_A.pdb").decode() == "HEADER 1\n"


def test_process_h5_file_subset_all_missing_returns_none(caplog, tmp_path):
    batch = tmp_path / "batch0"
    batch.mkdir()
    compress_and_save_h5(
        batch,
        (["only_A.pdb"], ["ATOM\n"], []),
    )
    h5 = batch / "pdbs.h5"
    out_dir = tmp_path / "staging"
    out_dir.mkdir()
    with caplog.at_level("WARNING"):
        zip_path = process_h5_file(
            str(h5),
            str(tmp_path),
            out_dir,
            pdb_codes=["missing_X"],
        )
    assert zip_path is None
    assert "not found" in caplog.text or "no matching" in caplog.text


def test_process_h5_file_empty_codes_returns_none(caplog, tmp_path):
    batch = tmp_path / "batch0"
    batch.mkdir()
    compress_and_save_h5(batch, (["a.pdb"], ["H\n"], []))
    with caplog.at_level("WARNING"):
        assert (
            process_h5_file(
                str(batch / "pdbs.h5"),
                str(tmp_path),
                tmp_path / "out",
                pdb_codes=["", "  "],
            )
            is None
        )


def test_merge_subset_shards_end_to_end(tmp_path):
    """Two shard zips from filtered reads merge to the union of members."""
    for i, (names, bodies) in enumerate(
        [
            (["a1.pdb", "a2.pdb"], ["HA1\n", "HA2\n"]),
            (["b1.pdb"], ["HB1\n"]),
        ]
    ):
        d = tmp_path / f"b{i}"
        d.mkdir()
        compress_and_save_h5(d, (names, bodies, []))
    out_staging = tmp_path / "stg"
    out_staging.mkdir()
    z1 = process_h5_file(
        str(tmp_path / "b0" / "pdbs.h5"),
        str(tmp_path),
        out_staging,
        pdb_codes=["a2.pdb"],
    )
    z2 = process_h5_file(
        str(tmp_path / "b1" / "pdbs.h5"),
        str(tmp_path),
        out_staging,
        pdb_codes=["b1"],
    )
    merged = tmp_path / "merged.zip"
    merge_shard_zips_flat([z1, z2], merged)
    with zipfile.ZipFile(merged, "r") as zf:
        assert sorted(zf.namelist()) == ["a2.pdb", "b1.pdb"]


def test_merge_shard_zips_flat_skips_duplicate_member(caplog, tmp_path):
    a = tmp_path / "a.zip"
    b = tmp_path / "b.zip"
    out = tmp_path / "merged.zip"
    _write_pdb_zip(a, {"same.pdb": "first"})
    _write_pdb_zip(b, {"same.pdb": "second"})
    with caplog.at_level("WARNING"):
        merge_shard_zips_flat([a, b], out)
    assert "Duplicate archive member" in caplog.text
    with zipfile.ZipFile(out, "r") as zf:
        assert zf.read("same.pdb") == b"first"
