"""Tests for input_path ID resolution (exact stem + AF model version / isoform)."""

from pathlib import Path

from toolbox.models.manage_dataset.extract_archive import (
    build_stem_to_paths,
    pick_single_path_for_canonical_id,
    resolve_id,
    retrieve_single_file,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_build_stem_to_paths_empty(tmp_path: Path) -> None:
    assert build_stem_to_paths(tmp_path) == {}


def test_resolve_exact_cif_only(tmp_path: Path) -> None:
    f = tmp_path / "Q5VSL9-2.cif"
    _touch(f)
    inv = build_stem_to_paths(tmp_path)
    out = resolve_id("Q5VSL9-2", inv)
    assert len(out) == 1
    assert out[0].resolve() == f.resolve()


def test_resolve_prefers_cif_over_pdb_same_stem(tmp_path: Path) -> None:
    cif = tmp_path / "Q5VSL9-2.cif"
    pdb = tmp_path / "Q5VSL9-2.pdb"
    _touch(pdb)
    _touch(cif)
    inv = build_stem_to_paths(tmp_path)
    out = resolve_id("Q5VSL9-2", inv)
    assert len(out) == 1
    assert out[0].suffix.lower() == ".cif"


def test_resolve_af_latest_version(tmp_path: Path) -> None:
    _touch(tmp_path / "AF-Q5VSL9-2-F1-model_v4.cif")
    v6 = tmp_path / "AF-Q5VSL9-2-F1-model_v6.cif"
    _touch(v6)
    inv = build_stem_to_paths(tmp_path)
    out = resolve_id("Q5VSL9-2", inv)
    assert len(out) == 1
    assert out[0].resolve() == v6.resolve()


def test_resolve_af_multi_fragment_same_version(tmp_path: Path) -> None:
    f1 = tmp_path / "AF-Q5VSL9-F1-model_v4.cif"
    f2 = tmp_path / "AF-Q5VSL9-F2-model_v4.cif"
    _touch(f1)
    _touch(f2)
    inv = build_stem_to_paths(tmp_path)
    out = resolve_id("Q5VSL9", inv)
    assert len(out) == 2
    assert {p.resolve() for p in out} == {f1.resolve(), f2.resolve()}


def test_pick_single_path_prefers_lowest_af_fragment(tmp_path: Path) -> None:
    f1 = tmp_path / "AF-Q5VSL9-F1-model_v4.cif"
    f2 = tmp_path / "AF-Q5VSL9-F2-model_v4.cif"
    _touch(f1)
    _touch(f2)
    chosen = pick_single_path_for_canonical_id([f2, f1])
    assert chosen.resolve() == f1.resolve()


def test_pick_single_path_non_af_tiebreak_by_path(tmp_path: Path) -> None:
    a = tmp_path / "a" / "x.cif"
    b = tmp_path / "b" / "y.cif"
    _touch(a)
    _touch(b)
    chosen = pick_single_path_for_canonical_id([b, a])
    assert chosen.resolve() == a.resolve()


def test_retrieve_single_file_canonical_pdb_code(tmp_path: Path) -> None:
    f = tmp_path / "AF-A0A2K6V5L6-F1-model_v6.cif"
    f.write_text("dummy", encoding="utf-8")
    data, pdb_code, ext = retrieve_single_file((str(f), "A0A2K6V5L6"))
    assert data == "dummy"
    assert pdb_code == "A0A2K6V5L6"
    assert ext == ".cif"


def test_retrieve_single_file_plain_path_uses_stem(tmp_path: Path) -> None:
    f = tmp_path / "AF-A0A2K6V5L6-F1-model_v6.cif"
    f.write_text("z", encoding="utf-8")
    _, pdb_code, _ = retrieve_single_file(str(f))
    assert pdb_code == "AF-A0A2K6V5L6-F1-model_v6"


def test_resolve_exact_wins_over_af_pattern(tmp_path: Path) -> None:
    exact = tmp_path / "Q5VSL9-2.cif"
    _touch(exact)
    _touch(tmp_path / "AF-Q5VSL9-2-F1-model_v6.cif")
    inv = build_stem_to_paths(tmp_path)
    out = resolve_id("Q5VSL9-2", inv)
    assert len(out) == 1
    assert out[0].resolve() == exact.resolve()


def test_resolve_loose_isoform_fallback(tmp_path: Path) -> None:
    iso = tmp_path / "AF-Q5VSL9-2-F1-model_v6.cif"
    _touch(iso)
    inv = build_stem_to_paths(tmp_path)
    out = resolve_id("Q5VSL9", inv)
    assert len(out) == 1
    assert out[0].resolve() == iso.resolve()


def test_resolve_af_exact_takes_precedence_over_loose_higher_version(tmp_path: Path) -> None:
    base_v4 = tmp_path / "AF-Q5VSL9-F1-model_v4.cif"
    iso_v6 = tmp_path / "AF-Q5VSL9-2-F1-model_v6.cif"
    _touch(base_v4)
    _touch(iso_v6)
    inv = build_stem_to_paths(tmp_path)
    out = resolve_id("Q5VSL9", inv)
    assert len(out) == 1
    assert out[0].resolve() == base_v4.resolve()


def test_resolve_missing_returns_empty(tmp_path: Path) -> None:
    _touch(tmp_path / "OTHER.cif")
    inv = build_stem_to_paths(tmp_path)
    assert resolve_id("Q5VSL9", inv) == []


def test_resolve_strips_id_not_applied_use_raw_match(tmp_path: Path) -> None:
    """Whitespace is stripped by caller in save_extracted_files; resolver is strict."""
    _touch(tmp_path / "Q5VSL9.cif")
    inv = build_stem_to_paths(tmp_path)
    assert resolve_id(" Q5VSL9", inv) == []
    assert resolve_id("Q5VSL9", inv) != []
