import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Union

from dask.distributed import as_completed
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from toolbox.models.manage_dataset.utils import read_all_pdbs_from_h5, read_pdbs_from_h5
from toolbox.scripts.create_archive.utils import archive_file_path, read_reversed_index
from toolbox.utlis.logging import logger


def _shard_zip_name(h5_file: str, data_path_str: str) -> str:
    """Stable unique .zip basename per H5 shard (avoids .../N/pdbs.h5 -> pdbs.zip collisions)."""
    h5_posix = Path(h5_file).as_posix()
    root = Path(data_path_str).as_posix().rstrip("/")
    if root and h5_posix.startswith(root + "/"):
        rel = h5_posix.removeprefix(root + "/")
    else:
        rel = h5_posix
    stem = rel.removesuffix(".h5").replace("/", "__")
    if len(stem) > 200:
        stem = hashlib.sha256(rel.encode()).hexdigest()[:24]
    return f"{stem}.zip"


def normalize_pdb_codes_for_h5(codes: list[str]) -> list[str]:
    """Expand dataset index ids for ``read_pdbs_from_h5`` matching."""
    out: list[str] = []
    seen: set[str] = set()
    for c in codes:
        s = (c or "").strip()
        if not s:
            continue
        stem = s.removesuffix(".pdb")
        for variant in (stem, f"{stem}.pdb"):
            if variant not in seen:
                seen.add(variant)
                out.append(variant)
    return out


def _requested_pdb_stems(codes: list[str]) -> set[str]:
    return {((c or "").strip()).removesuffix(".pdb") for c in codes if (c or "").strip()}


def process_h5_file(
    h5_file,
    data_path_str,
    output_dir,
    pdb_codes: Optional[list[str]] = None,
) -> Optional[str]:
    """Write one shard zip of PDB members from ``pdbs.h5``."""
    h5_path = Path(h5_file)
    archive_name = _shard_zip_name(h5_file, data_path_str)
    archive_path = Path(output_dir) / archive_name

    if pdb_codes is not None:
        normalized = normalize_pdb_codes_for_h5(pdb_codes)
        if not normalized:
            logger.warning("Empty pdb_codes for shard %s; skipping shard zip", h5_file)
            return None
        wanted = _requested_pdb_stems(pdb_codes)
        prots = read_pdbs_from_h5(str(h5_path), normalized)
        if prots is None:
            logger.error("Could not read PDB subset from H5 %s", h5_file)
            return None
        got_stems = {k.removesuffix(".pdb") for k in prots.keys()}
        missing_stems = sorted(wanted - got_stems)
        if missing_stems:
            logger.warning(
                "H5 %s: %s requested PDB(s) not found in shard (showing up to 40): %s",
                h5_file,
                len(missing_stems),
                missing_stems[:40] + (["..."] if len(missing_stems) > 40 else []),
            )
        if not prots:
            logger.warning(
                "H5 %s: no matching PDB content for requested codes; skipping shard zip",
                h5_file,
            )
            return None
    else:
        prots = read_all_pdbs_from_h5(str(h5_path))
        if prots is None:
            logger.error("Could not read PDBs from H5 %s", h5_file)
            return None
        if not prots:
            logger.warning("H5 %s is empty; skipping shard zip", h5_file)
            return None

    with zipfile.ZipFile(archive_path, "w") as zipf:
        for p, pdb_file_content in prots.items():
            code = p.removesuffix(".pdb")
            zipf.writestr(
                f"{code}.pdb",
                pdb_file_content,
                compress_type=zipfile.ZIP_DEFLATED,
            )

    return str(archive_path)


def merge_shard_zips_flat(
    shard_paths: Iterable[Union[str, Path]], dest_zip: Path
) -> None:
    """Merge shard zip archives into one flat zip of PDB members."""
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    ordered = sorted(Path(p).resolve() for p in shard_paths)

    with zipfile.ZipFile(dest_zip, "w") as zout:
        for sp in ordered:
            with zipfile.ZipFile(sp, "r") as zin:
                for info in zin.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename
                    if name in seen:
                        logger.warning(
                            "Duplicate archive member %s (skipping from %s)",
                            name,
                            sp,
                        )
                        continue
                    seen.add(name)
                    data = zin.read(name)
                    zout.writestr(
                        name,
                        data,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )


def create_structures_archive(
    structures_dataset: "StructuresDataset",
    output_dir: Path,
    timestamp: str,
    keep_shard_zips: bool = False,
) -> Optional[Path]:
    data_path = structures_dataset.config.data_path
    dataset_path = structures_dataset.dataset_path()
    proteins_index = read_reversed_index(structures_dataset, "dataset")

    staging = output_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    client = structures_dataset._client

    futures = []
    for h5_file, ids in proteins_index.items():
        if not ids:
            logger.warning(
                "No protein IDs listed for shard %s in dataset_reversed.idx; skipping",
                h5_file,
            )
            continue
        futures.append(
            client.submit(process_h5_file, h5_file, data_path, staging, ids)
        )

    n = len(futures)
    logger.info(
        "Building %s PDB shard zip(s) in staging, then one merged archive under %s",
        n,
        output_dir,
    )

    shard_paths: list[str] = []
    with logging_redirect_tqdm():
        with tqdm(total=n, desc="H5 shards -> zip", unit="h5") as pbar:
            for fut in as_completed(futures):
                path = fut.result()
                if path:
                    shard_paths.append(path)
                pbar.update(1)

    if not shard_paths:
        logger.error(
            "No PDB shard zips produced (missing index, empty id lists, or no matching "
            "PDBs in H5). Check %s",
            Path(dataset_path) / "dataset_reversed.idx",
        )
        if not keep_shard_zips and staging.exists():
            shutil.rmtree(staging)
        return None

    final_archive_path = archive_file_path(
        output_dir,
        "pdb",
        structures_dataset.dataset_dir_name(),
        timestamp,
        "zip",
    )
    merge_shard_zips_flat(shard_paths, final_archive_path)
    logger.info("Wrote merged PDB archive %s", final_archive_path)

    if not keep_shard_zips:
        shutil.rmtree(staging)

    return final_archive_path
