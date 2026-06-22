import hashlib
import os
import pickle
import time
from contextlib import nullcontext
from typing import Iterable, List, NamedTuple, Optional, Tuple, Dict, Union
import zipfile
import tarfile
from pathlib import Path

from dask.distributed import worker_client
from toolbox.models.manage_dataset.index.handle_index import add_new_files_to_index, create_index
from toolbox.models.utils.create_client import total_workers

from toolbox.models.manage_dataset.compute_batches import ComputeBatches
from toolbox.models.manage_dataset.create_dataset_timing import get_active_timings
from toolbox.models.manage_dataset.utils import (
    compress_and_save_h5,
    mkdir_for_batches,
    format_time
)
from toolbox.models.utils.cif2pdb import cif_to_pdb

from toolbox.utlis.logging import logger
import re


def extract_archive(
    input_path, structures_dataset: "StructureDataset"
) -> Optional[Path]:

    if not input_path.exists():
        logger.error(f"Error: The provided path {input_path} does not exist.")
        return None

    if is_archive(input_path):
        logger.info(f"Processing archive to extract protein files: {input_path}")
        extracted_path = structures_dataset.dataset_repo_path() / "extracted_files"
        os.makedirs(extracted_path, exist_ok=True)

        if zipfile.is_zipfile(input_path):
            logger.debug("Extracting zip file")
            with zipfile.ZipFile(input_path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)
        elif tarfile.is_tarfile(input_path):
            logger.debug("Extracting tar/tar.gz file")
            with tarfile.open(
                input_path, "r:*"
            ) as tar_ref:  # 'r:*' auto-detects compression
                tar_ref.extractall(extracted_path)
        else:
            logger.error("Provided path is neither a directory nor a supported archive.")
            return None

        return extracted_path

    return input_path  # Return original path if it's not an archive


def is_archive(path):
    if os.path.isdir(path):
        return False
    # Check if the file is a zip or tar/tar.gz archive
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


_STEM_CACHE_VERSION = 1
_STEM_CACHE_ENV_DISABLE = "DEEPFRI_DISABLE_STEM_CACHE"
_STEM_CACHE_RECOUNT_MAX_FILES = 10_000


def _stem_cache_enabled() -> bool:
    return os.environ.get(_STEM_CACHE_ENV_DISABLE, "").lower() not in (
        "1",
        "true",
        "yes",
    )


def _stem_cache_dir() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home) if cache_home else Path.home() / ".cache"
    return root / "deepfri" / "stem_to_paths"


def _stem_cache_file(base: Path) -> Path:
    digest = hashlib.sha256(str(base).encode()).hexdigest()
    return _stem_cache_dir() / f"{digest}.pkl"


def _scan_stem_to_paths(
    base: Path,
) -> Tuple[Dict[str, List[str]], List[Tuple[str, int, int]]]:
    """Walk base and return stem index (string paths) plus per-directory metadata."""
    stem_to_paths: Dict[str, List[str]] = {}
    dir_fingerprints: List[Tuple[str, int, int]] = []
    base_str = str(base)
    for dirpath, _, filenames in os.walk(base, followlinks=False):
        rel = os.path.relpath(dirpath, base_str)
        if rel == ".":
            rel = ""
        n_structure_files = sum(
            1
            for name in filenames
            if name.lower().endswith(".cif") or name.lower().endswith(".pdb")
        )
        try:
            mtime_ns = os.stat(dirpath, follow_symlinks=False).st_mtime_ns
        except OSError:
            return {}, []
        dir_fingerprints.append((rel, mtime_ns, n_structure_files))
        for name in filenames:
            low = name.lower()
            if not (low.endswith(".cif") or low.endswith(".pdb")):
                continue
            p = Path(dirpath) / name
            stem_to_paths.setdefault(p.stem, []).append(str(p))
    dir_fingerprints.sort()
    return stem_to_paths, dir_fingerprints


def _dir_fingerprints_valid(
    base: Path, fingerprints: List[Tuple[str, int, int]]
) -> bool:
    stored_dirs = {rel for rel, _, _ in fingerprints}
    try:
        with os.scandir(base) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if entry.name not in stored_dirs:
                    return False
    except OSError:
        return False

    for rel, mtime_ns, n_structure_files in fingerprints:
        path = base if not rel else base / rel
        try:
            st = os.stat(path, follow_symlinks=False)
        except OSError:
            return False
        if st.st_mtime_ns != mtime_ns:
            return False
        if n_structure_files <= _STEM_CACHE_RECOUNT_MAX_FILES:
            with os.scandir(path) as it:
                actual = sum(
                    1
                    for entry in it
                    if entry.is_file(follow_symlinks=False)
                    and entry.name.lower().endswith((".cif", ".pdb"))
                )
            if actual != n_structure_files:
                return False
    return True


def _load_stem_cache(base: Path) -> Optional[Dict[str, List[str]]]:
    cache_file = _stem_cache_file(base)
    if not cache_file.is_file():
        return None
    try:
        with cache_file.open("rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.UnpicklingError, KeyError, TypeError, ValueError):
        return None
    if payload.get("version") != _STEM_CACHE_VERSION:
        return None
    if payload.get("base") != str(base):
        return None
    fingerprints = payload.get("dir_fingerprints")
    stem_to_paths = payload.get("stem_to_paths")
    if not isinstance(fingerprints, list) or not isinstance(stem_to_paths, dict):
        return None
    if not _dir_fingerprints_valid(base, fingerprints):
        return None
    return stem_to_paths


def _save_stem_cache(
    base: Path,
    stem_to_paths: Dict[str, List[str]],
    dir_fingerprints: List[Tuple[str, int, int]],
) -> None:
    cache_file = _stem_cache_file(base)
    payload = {
        "version": _STEM_CACHE_VERSION,
        "base": str(base),
        "dir_fingerprints": dir_fingerprints,
        "stem_to_paths": stem_to_paths,
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_file)
    except OSError as exc:
        logger.debug(f"Could not write stem index cache {cache_file}: {exc}")


def _paths_from_stem_cache(stem_to_paths: Dict[str, List[str]]) -> Dict[str, List[Path]]:
    return {
        stem: [Path(path_str) for path_str in paths]
        for stem, paths in stem_to_paths.items()
    }


def build_stem_to_paths(extracted_path: Path, *, use_cache: bool = True) -> Dict[str, List[Path]]:
    """
    Collect all .pdb / .cif under extracted_path, keyed by filename stem.
    Paths are absolute for reliable opening (archive vs directory).

    Results are cached on disk (XDG cache dir) keyed by the resolved input path.
    Invalidation uses directory mtimes from the last scan. Set
    ``DEEPFRI_DISABLE_STEM_CACHE=1`` or ``use_cache=False`` to skip the cache.
    """
    base = extracted_path.resolve()
    if not base.is_dir():
        return {}

    cache_on = use_cache and _stem_cache_enabled()
    if cache_on:
        cached = _load_stem_cache(base)
        if cached is not None:
            logger.debug(
                f"Loaded stem index from cache for {base} ({len(cached)} stems)"
            )
            return _paths_from_stem_cache(cached)

    stem_to_paths, dir_fingerprints = _scan_stem_to_paths(base)
    if cache_on and dir_fingerprints:
        _save_stem_cache(base, stem_to_paths, dir_fingerprints)
        logger.debug(f"Saved stem index cache for {base} ({len(stem_to_paths)} stems)")

    return _paths_from_stem_cache(stem_to_paths)


def _pick_path_prefer_cif(paths: List[Path]) -> Path:
    """If both .cif and .pdb exist for the same stem, prefer .cif."""
    cifs = [p for p in paths if p.suffix.lower() == ".cif"]
    if cifs:
        return sorted(cifs, key=lambda x: str(x))[0]
    return sorted(paths, key=lambda x: str(x))[0]


# Parse an AlphaFold stem into (accession, fragment, version) in one pass.
_AF_STEM_RE = re.compile(r"^AF-(.+)-F(\d+)-model_v(\d+)$")
# Trailing "-<digits>" of an accession marks an isoform suffix (loose match base).
_TRAILING_ISOFORM_RE = re.compile(r"-\d+$")


class StemResolutionIndex(NamedTuple):
    """
    Pre-computed lookup tables that turn per-id resolution into O(1) dict lookups
    instead of scanning every stem with a fresh regex for each requested id.

    - ``picked_by_stem``: stem -> single chosen path (prefer .cif over .pdb).
    - ``af_exact``: accession -> [(version, stem)] for ``AF-{accession}-F*-model_v*``.
    - ``af_loose``: isoform base -> [(version, stem)] where accession == ``{base}-{digits}``.
    """

    stem_to_paths: Dict[str, List[Path]]
    picked_by_stem: Dict[str, Path]
    af_exact: Dict[str, List[Tuple[int, str]]]
    af_loose: Dict[str, List[Tuple[int, str]]]


def build_resolution_index(
    stem_to_paths: Dict[str, List[Path]],
    picked_by_stem: Optional[Dict[str, Path]] = None,
) -> StemResolutionIndex:
    """
    Index all stems once so requested ids can be resolved by dict lookup.

    This replaces the previous approach of compiling and running two regexes
    against every stem for each requested id (O(ids * stems)) with a single
    pass over the stems (O(stems)) plus O(1) lookups per id.
    """
    if picked_by_stem is None:
        picked_by_stem = {
            stem: _pick_path_prefer_cif(paths) for stem, paths in stem_to_paths.items()
        }

    af_exact: Dict[str, List[Tuple[int, str]]] = {}
    af_loose: Dict[str, List[Tuple[int, str]]] = {}
    for stem in stem_to_paths:
        m = _AF_STEM_RE.match(stem)
        if not m:
            continue
        accession = m.group(1)
        version = int(m.group(3))
        af_exact.setdefault(accession, []).append((version, stem))
        iso = _TRAILING_ISOFORM_RE.search(accession)
        if iso:
            base = accession[: iso.start()]
            af_loose.setdefault(base, []).append((version, stem))

    return StemResolutionIndex(stem_to_paths, picked_by_stem, af_exact, af_loose)


def _max_version_paths(
    group: List[Tuple[int, str]], picked_by_stem: Dict[str, Path]
) -> List[Path]:
    """
    From [(version, stem)], pick the globally highest version and return one path
    per stem at that version (sorted by stem for deterministic ordering).
    """
    if not group:
        return []
    max_v = max(v for v, _ in group)
    stems_at_max = sorted({stem for v, stem in group if v == max_v})
    return [picked_by_stem[s] for s in stems_at_max]


def resolve_id_with_index(requested_id: str, index: StemResolutionIndex) -> List[Path]:
    """
    Resolve a requested protein id against a pre-built ``StemResolutionIndex``.

    1) Exact stem match (prefer .cif over .pdb).
    2) AF exact: AF-{id}-F{N}-model_v{V} — highest V, all fragments at that V.
    3) AF loose (isoform): AF-{id}-{digits}-F{N}-model_v{V} — same version rule.
    """
    if requested_id in index.stem_to_paths:
        return [index.picked_by_stem[requested_id]]

    exact = index.af_exact.get(requested_id)
    if exact:
        return _max_version_paths(exact, index.picked_by_stem)

    loose = index.af_loose.get(requested_id)
    if loose:
        return _max_version_paths(loose, index.picked_by_stem)

    return []


def resolve_id(requested_id: str, stem_to_paths: Dict[str, List[Path]]) -> List[Path]:
    """
    Resolve a requested protein id to file path(s) under input_path / extracted tree.

    Convenience wrapper that builds a one-off index; for resolving many ids against
    the same stem set, build a :class:`StemResolutionIndex` once and call
    :func:`resolve_id_with_index`.
    """
    return resolve_id_with_index(requested_id, build_resolution_index(stem_to_paths))


_AF_FRAGMENT_NUM_RE = re.compile(r"-F(\d+)-model_v\d+$")


def pick_single_path_for_canonical_id(paths: List[Path]) -> Path:
    """
    When multiple structures match one ids_file id (e.g. F1 and F2 at same model version),
    keep a single file: lowest AF fragment number F{N}; ties by resolved path string.
    Non-AF stems sort after AF (fragment key 10**9).
    """
    if not paths:
        raise ValueError("paths must be non-empty")
    if len(paths) == 1:
        return paths[0]
    scored: List[Tuple[int, str, Path]] = []
    for p in paths:
        stem = p.stem
        m = _AF_FRAGMENT_NUM_RE.search(stem)
        frag = int(m.group(1)) if m else 10**9
        scored.append((frag, str(p.resolve()), p))
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def save_extracted_files(
    structures_dataset: "StructuresDataset",
    extracted_path: Path,
    ids: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """
    Process extracted files from archive and return missing IDs.
    
    Returns:
        List of missing IDs (processed for AFDB if needed), or None if ids was None
    """
    Path(structures_dataset.structures_path()).mkdir(exist_ok=True, parents=True)
    pdb_repo_path = structures_dataset.structures_path()

    extracted_path = Path(extracted_path)
    timings = get_active_timings()
    if timings is not None:
        with timings.measure("build_stem_to_paths"):
            stem_to_paths = build_stem_to_paths(extracted_path)
    else:
        stem_to_paths = build_stem_to_paths(extracted_path)
    picked_by_stem = {
        stem: _pick_path_prefer_cif(paths) for stem, paths in stem_to_paths.items()
    }

    logger.debug(f"extracted files: {len(stem_to_paths)} unique stems")

    canonical_base_to_source_name: Dict[str, str] = {}
    use_canonical_ids = ids is not None

    if ids is None:
        files = [str(picked_by_stem[s]) for s in sorted(picked_by_stem)]
        chunks = list(structures_dataset.chunk(files))
        missing_files = None
    else:
        logger.info(
            f"Searching for {len(ids)} ids among {len(stem_to_paths)} stems under {extracted_path}"
        )

        wanted_items: List[Union[str, Tuple[str, str]]] = []
        seen_resolved: set[str] = set()
        missing_files = []

        resolve_ctx = timings.measure("resolve_requested_ids") if timings else nullcontext()
        with resolve_ctx:
            resolution_index = build_resolution_index(stem_to_paths, picked_by_stem)
            for raw_id in ids:
                rid = raw_id.strip()
                resolved_paths = resolve_id_with_index(rid, resolution_index)
                if not resolved_paths:
                    missing_files.append(raw_id)
                    continue
                chosen = pick_single_path_for_canonical_id(resolved_paths)
                key = str(chosen.resolve())
                if key not in seen_resolved:
                    seen_resolved.add(key)
                    wanted_items.append((key, rid))
                    canonical_base_to_source_name[rid] = chosen.name

        logger.info(
            f"Resolved {len(wanted_items)} file path(s), {len(missing_files)} missing "
            f"out of {len(ids)} requested ids"
        )
        chunks = list(structures_dataset.chunk(wanted_items))

    if timings is not None:
        with timings.measure("mkdir_for_batches"):
            mkdir_for_batches(pdb_repo_path, len(chunks))
    else:
        mkdir_for_batches(pdb_repo_path, len(chunks))

    new_files_index = {}

    def run(input_data, machine):
        return structures_dataset._client.submit(
            retrieve_protein_file_to_h5, *input_data, [machine], workers=[machine]
        )

    def collect(result):
        if len(result) == 3:
            downloaded_pdbs, file_path, worker_timings = result
            if timings is not None:
                timings.merge_worker_timings(worker_timings)
        else:
            downloaded_pdbs, file_path = result
        new_files_index.update({k: file_path for k in downloaded_pdbs})

    compute_batches = ComputeBatches(
        structures_dataset._client, run, collect, "pdb_extracted_from_archive", len(chunks)
    )

    inputs = ((pdb_repo_path / f"{i}", ids_chunk) for i, ids_chunk in enumerate(chunks))

    logger.info(f"Extracting PDBs from archive {extracted_path} in {len(chunks)} batches")

    factor = 1
    factor = 15 if total_workers() > 1500 else factor
    factor = 20 if total_workers() > 2000 else factor
    compute_batches.compute(inputs, factor=factor)

    logger.info("Adding new files to index")

    no_chain_to_chain_dict = {}
    for id_with_chain in new_files_index.keys():
        # Remove last '_' and everything after it to get id_without_chain
        id_without_chain = id_with_chain.rsplit('_', 1)[0]
        no_chain_to_chain_dict[id_without_chain] = id_with_chain

    input_structures_index = {}

    if use_canonical_ids:
        for id_with_chain in sorted(new_files_index.keys()):
            base = id_with_chain.rsplit("_", 1)[0]
            src = canonical_base_to_source_name.get(base)
            if src is not None:
                input_structures_index[id_with_chain] = src
    else:
        for stem, picked in sorted(picked_by_stem.items(), key=lambda x: x[0]):
            file_path = picked
            id_with_chain = no_chain_to_chain_dict.get(stem, None)
            if id_with_chain is None:
                continue
            input_structures_index[id_with_chain] = file_path.name

    try:
        if timings is not None:
            with timings.measure("update_dataset_index"):
                add_new_files_to_index(
                    structures_dataset.dataset_index_file_path(),
                    new_files_index,
                    structures_dataset.config.data_path,
                )
            with timings.measure("update_input_structures_index"):
                create_index(
                    structures_dataset.input_structures_index_path(),
                    input_structures_index,
                    structures_dataset.config.data_path,
                )
        else:
            add_new_files_to_index(
                structures_dataset.dataset_index_file_path(),
                new_files_index,
                structures_dataset.config.data_path,
            )
            create_index(
                structures_dataset.input_structures_index_path(),
                input_structures_index,
                structures_dataset.config.data_path,
            )
    except Exception as e:
        logger.error(f"Failed to update index: {e}")
    
    return missing_files


def retrieve_protein_file_to_h5(
    path_for_batch: Path,
    pdb_ids: Iterable[Union[str, Tuple[str, str]]],
    workers: List[str] = None,
) -> Tuple[List[str], str, Dict[str, float]]:
    with worker_client() as client:
        start_time = time.perf_counter()

        read_futures = client.map(retrieve_single_file, pdb_ids, workers=workers)
        converted_pdb_futures = client.map(file_to_pdb, read_futures, workers=workers)
        download_start_time = time.time()
        aggregated_future = client.submit(
            aggregate_results,
            converted_pdb_futures,
            download_start_time,
            workers=workers,
        )
        aggregated = client.gather(aggregated_future)
        pipeline_time = time.perf_counter() - start_time

        h5_start = time.perf_counter()
        h5_file_path = client.submit(
            compress_and_save_h5,
            path_for_batch,
            aggregated,
            pure=False,
            workers=workers,
        ).result()
        pdb_ids_out = aggregated[0]
        h5_time = time.perf_counter() - h5_start

        total_time = pipeline_time + h5_time
        logger.info(
            f"Total processing time {path_for_batch.stem}: {format_time(total_time)} "
            f"(extract {format_time(pipeline_time)}, h5 {format_time(h5_time)})"
        )

        worker_timings = {
            "protein_extraction_pipeline": pipeline_time,
            "h5_save": h5_time,
        }
        return pdb_ids_out, h5_file_path, worker_timings


def retrieve_single_file(
    item: Union[str, Tuple[str, str], List[str]],
):
    """
    Load structure file for conversion.

    ``item`` is either a path string, or ``(path_str, canonical_pdb_code)`` where
    ``canonical_pdb_code`` is the ids_file token used as ``cif_to_pdb`` / PDB key base
    (e.g. UniProt accession), not the AF CIF filename stem.
    """
    canonical: Optional[str] = None
    if isinstance(item, (tuple, list)) and len(item) == 2:
        file_path = Path(item[0])
        canonical = str(item[1])
    else:
        file_path = Path(item)
    pdb_code = canonical if canonical is not None else file_path.stem
    file_extension = file_path.suffix
    with open(file_path, "r") as file:
        return file.read(), pdb_code, file_extension


def file_to_pdb(input_data):
    file_data, file_name, file_extension = input_data
    if file_extension == ".cif":
        return cif_to_pdb(file_data, file_name)
    elif file_extension == ".pdb":
        return {f"{file_name}": str(file_data)}
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def aggregate_results(
    protein_pdbs_with_cif: List[Dict[str, str]], download_start_time: float
) -> Tuple[List[str], List[str]]:
    end_time = time.time()

    logger.debug(f"Download time: {format_time(end_time - download_start_time)}")

    all_res_pdbs = []
    all_contents = []

    for prot in protein_pdbs_with_cif:
        all_res_pdbs.extend(prot.keys())
        all_contents.extend(prot.values())

    return all_res_pdbs, all_contents
