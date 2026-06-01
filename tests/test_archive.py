import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from fridata import create_parser
from toolbox.models.manage_dataset.index.handle_index import create_index
from toolbox.models.manage_dataset.utils import compress_and_save_h5
from toolbox.scripts.archive import (
    _shard_zip_name,
    create_archive,
    merge_shard_zips_flat,
    normalize_pdb_codes_for_h5,
    process_h5_file,
)
from toolbox.scripts.create_archive.numeric import create_numeric_archive
from toolbox.scripts.create_archive.utils import normalize_archive_types


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


class _FakeConfig:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.separator = "-"


class _FakeDataset:
    def __init__(self, data_path: Path, name: str = "PDB-subset--test"):
        self.config = _FakeConfig(str(data_path))
        self._name = name
        self._dataset_path = data_path / "datasets" / name
        self._dataset_path.mkdir(parents=True, exist_ok=True)

    def dataset_path(self):
        return self._dataset_path

    def dataset_dir_name(self):
        return self._name


def _write_embeddings_batch(path: Path, entries: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        for protein_id, array in entries.items():
            hf.create_dataset(protein_id, data=array)


def _write_coordinates_batch(path: Path, entries: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        for protein_id, array in entries.items():
            group = hf.create_group(protein_id)
            group.create_dataset("coords", data=array)


def _write_distograms_batch(path: Path, entries: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        for protein_id, array in entries.items():
            group = hf.create_group(protein_id)
            group.create_dataset("distogram", data=array)


def test_normalize_archive_types_accepts_all_and_comma_separated_values():
    assert normalize_archive_types("all") == {
        "coordinates",
        "distograms",
        "embeddings",
        "structures",
    }
    assert normalize_archive_types("structures,embeddings") == {
        "structures",
        "embeddings",
    }


def test_cli_create_archive_type_parsing_accepts_multiple_values(tmp_path):
    parser = create_parser()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    args = parser.parse_args(
        [
            "create_archive",
            "-p",
            str(dataset_path),
            "-t",
            "structures,embeddings",
        ]
    )
    assert args.command == "create_archive"
    assert args.type == ["embeddings", "structures"]


def test_cli_create_archive_type_parsing_rejects_unknown_value(tmp_path):
    parser = create_parser()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "create_archive",
                "-p",
                str(dataset_path),
                "-t",
                "structures,unknown",
            ]
        )


def test_create_numeric_archive_exports_embeddings_npz(tmp_path):
    ds = _FakeDataset(tmp_path)
    h5_path = tmp_path / "embeddings" / ds.dataset_dir_name() / "batch_0.h5"
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = np.ones((3, 2), dtype=np.float32)
    _write_embeddings_batch(h5_path, {"protA": first, "protB": second})
    create_index(
        ds.dataset_path() / "embeddings.idx",
        {"protA": str(h5_path), "protB": str(h5_path)},
        ds.config.data_path,
    )

    out = create_numeric_archive(
        ds,
        tmp_path / "archives" / ds.dataset_dir_name(),
        "20260102030405",
        "embeddings",
    )

    assert out is not None
    assert out.name == f"archive_embeddings_{ds.dataset_dir_name()}_20260102030405.npz"
    loaded = np.load(out)
    assert loaded["__protein_ids__"].tolist() == ["protA", "protB"]
    np.testing.assert_array_equal(loaded["arr_00000000"], first)
    np.testing.assert_array_equal(loaded["arr_00000001"], second)


def test_create_numeric_archive_exports_coordinates_and_distograms(tmp_path):
    ds = _FakeDataset(tmp_path)
    coords_h5 = tmp_path / "coordinates" / ds.dataset_dir_name() / "batch_0_ca.h5"
    dist_h5 = tmp_path / "distograms" / ds.dataset_dir_name() / "batch_0.h5"
    coords = np.array([[1, 1.0, 2.0, 3.0]], dtype=np.float32)
    dist = np.array([[0.0, 4.0], [4.0, 0.0]], dtype=np.float32)
    _write_coordinates_batch(coords_h5, {"protA": coords})
    _write_distograms_batch(dist_h5, {"protA": dist})
    create_index(
        ds.dataset_path() / "coordinates.idx",
        {"protA": str(coords_h5)},
        ds.config.data_path,
    )
    create_index(
        ds.dataset_path() / "distograms.idx",
        {"protA": str(dist_h5)},
        ds.config.data_path,
    )

    out_dir = tmp_path / "archives" / ds.dataset_dir_name()
    coords_out = create_numeric_archive(ds, out_dir, "20260102030405", "coordinates")
    dist_out = create_numeric_archive(ds, out_dir, "20260102030405", "distograms")

    assert coords_out is not None
    assert dist_out is not None
    np.testing.assert_array_equal(np.load(coords_out)["arr_00000000"], coords)
    np.testing.assert_array_equal(np.load(dist_out)["arr_00000000"], dist)


def test_create_archive_numeric_outputs_share_timestamp(tmp_path):
    ds = _FakeDataset(tmp_path)
    emb_h5 = tmp_path / "embeddings" / ds.dataset_dir_name() / "batch_0.h5"
    coords_h5 = tmp_path / "coordinates" / ds.dataset_dir_name() / "batch_0_ca.h5"
    _write_embeddings_batch(emb_h5, {"protA": np.ones((2, 2), dtype=np.float32)})
    _write_coordinates_batch(
        coords_h5, {"protA": np.ones((1, 4), dtype=np.float32)}
    )
    create_index(
        ds.dataset_path() / "embeddings.idx",
        {"protA": str(emb_h5)},
        ds.config.data_path,
    )
    create_index(
        ds.dataset_path() / "coordinates.idx",
        {"protA": str(coords_h5)},
        ds.config.data_path,
    )

    outputs = create_archive(ds, archive_types=["embeddings", "coordinates"])

    assert set(outputs) == {"embeddings", "coordinates"}
    timestamps = {path.stem.rsplit("_", 1)[-1] for path in outputs.values()}
    assert len(timestamps) == 1
