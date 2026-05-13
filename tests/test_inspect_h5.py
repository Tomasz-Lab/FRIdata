"""inspect_h5 batch layouts: contracts for render_inspect_h5_content (TDD)."""

import zlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from fridata import create_parser, validate_inspect_h5_cli_args


def write_distograms_batch(path: Path, entries: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as hf:
        for key, matrix in entries.items():
            g = hf.create_group(key)
            g.create_dataset("distogram", data=matrix)


def write_coordinates_batch(path: Path, entries: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as hf:
        for key, coords in entries.items():
            g = hf.create_group(key)
            g.create_dataset("coords", data=coords)


def write_embeddings_batch(path: Path, entries: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as hf:
        for key, emb in entries.items():
            hf.create_dataset(key, data=emb)


def write_pdbs_shard(path: Path, codes_to_text: dict[str, str]) -> None:
    codes = sorted(codes_to_text.keys())
    texts = [codes_to_text[k] for k in codes]
    raw = zlib.compress("|".join(texts).encode("utf-8"))
    with h5py.File(path, "w") as hf:
        grp = hf.create_group("files")
        grp.create_dataset(";".join(codes), data=np.frombuffer(raw, dtype=np.uint8))


@pytest.mark.parametrize(
    "mode,file_writer,key_line,shape_line",
    [
        (
            "distogram",
            lambda p: write_distograms_batch(
                p,
                {
                    "protA": np.array([[1.0, 2], [3, 4]], dtype=np.float32),
                    "protB": np.array([[10.0]], dtype=np.float32),
                },
            ),
            "=== protA ===",
            "shape: (2, 2)",
        ),
        (
            "coordinates",
            lambda p: write_coordinates_batch(
                p,
                {
                    "p1": np.array([[0, 1, 2, 3]], dtype=np.float32),
                    "p2": np.ones((2, 4), dtype=np.float32),
                },
            ),
            "=== p1 ===",
            "shape: (1, 4)",
        ),
        (
            "embedding",
            lambda p: write_embeddings_batch(
                p, {"seq_one": np.zeros((5, 3), dtype=np.float32)}
            ),
            "=== seq_one ===",
            "shape: (5, 3)",
        ),
    ],
)
def test_render_numeric_modes_include_section_shape_preview(
    tmp_path: Path,
    mode: str,
    file_writer,
    key_line: str,
    shape_line: str,
) -> None:
    from toolbox.utlis.inspect_h5 import render_inspect_h5_content

    h5_path = tmp_path / "batch.h5"
    file_writer(h5_path)
    text = render_inspect_h5_content(h5_path, mode, names=None)
    assert text is not None
    assert key_line in text
    assert shape_line in text
    assert "preview:" in text


def test_render_distogram_ordered_name_filter(tmp_path: Path) -> None:
    from toolbox.utlis.inspect_h5 import render_inspect_h5_content

    h5_path = tmp_path / "b.h5"
    write_distograms_batch(
        h5_path,
        {
            "z_last": np.eye(2, dtype=np.float32),
            "a_first": np.ones((2, 2), dtype=np.float32),
        },
    )
    text = render_inspect_h5_content(h5_path, "distogram", names=("z_last", "a_first"))
    assert text is not None
    assert text.index("=== z_last ===") < text.index("=== a_first ===")


def test_render_large_distogram_skips_high_cost_stats(tmp_path: Path) -> None:
    """Matrices over FULL_STATS_MAX_ELEMENTS omit full-volume stats line."""
    from toolbox.utlis import inspect_h5 as ih

    path = tmp_path / "big.h5"
    n = 400
    rng = np.random.default_rng(0)
    write_distograms_batch(path, {"bigprot": rng.random((n, n)).astype(np.float32)})
    assert n * n > ih.FULL_STATS_MAX_ELEMENTS

    text = ih.render_inspect_h5_content(path, "distogram", None)
    assert text is not None
    assert "=== bigprot ===" in text
    assert "preview:" in text
    assert "full-volume statistics omitted" in text


def test_render_missing_name_warns_and_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from toolbox.utlis.inspect_h5 import render_inspect_h5_content

    path = tmp_path / "m.h5"
    write_distograms_batch(path, {"have": np.eye(2, dtype=np.float32)})
    with caplog.at_level("WARNING"):
        text = render_inspect_h5_content(path, "distogram", names=("ghost", "have"))
    assert text is not None
    assert "=== have ===" in text
    assert "ghost" in caplog.text


def test_render_skips_bad_group_but_keeps_good(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from toolbox.utlis.inspect_h5 import render_inspect_h5_content

    p = tmp_path / "partial.h5"
    with h5py.File(p, "w") as hf:
        g = hf.create_group("good")
        g.create_dataset("distogram", data=np.ones((2, 2), dtype=np.float32))
        hf.create_group("bad_empty")

    with caplog.at_level("WARNING"):
        text = render_inspect_h5_content(p, "distogram", None)
    assert text is not None
    assert "=== good ===" in text
    assert "bad_empty" in caplog.text


def test_render_content_pdbs_shard(tmp_path: Path) -> None:
    from toolbox.utlis.inspect_h5 import render_inspect_h5_content

    p = tmp_path / "pdbs.h5"
    write_pdbs_shard(p, {"CODE1": "HEADER\nLINE2", "CODE2": "OTHER"})
    text = render_inspect_h5_content(p, "content", None)
    assert text is not None
    assert "=== CODE1 ===" in text
    assert "HEADER" in text

    sub = render_inspect_h5_content(p, "content", names=("CODE2",))
    assert sub is not None
    assert "OTHER" in sub


def test_render_missing_file_returns_none(tmp_path: Path) -> None:
    from toolbox.utlis.inspect_h5 import render_inspect_h5_content

    missing = tmp_path / "does_not_exist.h5"
    assert render_inspect_h5_content(missing, "structure", None) is None


def test_cli_inspect_accepts_numeric_modes_and_name(tmp_path: Path) -> None:
    parser = create_parser()
    h5_path = tmp_path / "any.h5"
    h5_path.touch()
    args = parser.parse_args(
        ["inspect_h5", "--mode", "distogram", "--name", "a,b", str(h5_path)]
    )
    validate_inspect_h5_cli_args(args, parser)
    assert args.command == "inspect_h5"
    assert args.mode == "distogram"
    assert args.pdb_names == ["a", "b"]

    args2 = parser.parse_args(["inspect_h5", "--mode", "embedding", str(h5_path)])
    validate_inspect_h5_cli_args(args2, parser)
    assert args2.mode == "embedding"


def test_cli_name_rejected_for_structure_mode(tmp_path: Path) -> None:
    parser = create_parser()
    h5_path = tmp_path / "x.h5"
    h5_path.touch()
    args = parser.parse_args(
        ["inspect_h5", "--mode", "structure", "--name", "only", str(h5_path)]
    )
    with pytest.raises(SystemExit):
        validate_inspect_h5_cli_args(args, parser)
