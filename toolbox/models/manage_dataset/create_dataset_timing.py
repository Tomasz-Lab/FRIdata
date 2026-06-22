"""Timing collection and reporting for StructuresDataset.create_dataset."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

from toolbox.models.manage_dataset.utils import format_time
from toolbox.utlis.logging import logger


# Phase -> high-level category for summary analysis
PHASE_CATEGORIES: Dict[str, str] = {
    "client_setup": "setup",
    "requested_ids": "setup",
    "read_indexes": "index",
    "find_present_and_missing": "index",
    "create_index_present": "index",
    "read_index_final": "index",
    "update_dataset_index": "index",
    "update_input_structures_index": "index",
    "update_downloaded_structures_index": "index",
    "extract_archive": "input",
    "build_stem_to_paths": "index",
    "resolve_requested_ids": "index",
    "mkdir_for_batches": "file_save",
    "batch_orchestration": "orchestration",
    "protein_extraction_pipeline": "protein_extraction",
    "protein_read_convert_pipeline": "protein_extraction",
    "h5_save": "file_save",
    "save_dataset_metadata": "file_save",
    "download_ids_total": "orchestration",
    "write_missing_ids_file": "file_save",
}


CATEGORY_LABELS = {
    "index": "Index processing",
    "protein_extraction": "Protein extraction",
    "file_save": "File saving",
    "input": "Input / archive",
    "setup": "Setup",
    "orchestration": "Orchestration (Dask batches)",
}


@dataclass
class CreateDatasetTimings:
    """Accumulates wall-clock seconds per named phase during create_dataset."""

    _seconds: Dict[str, float] = field(default_factory=dict)
    _counts: Dict[str, int] = field(default_factory=dict)

    def add(self, phase: str, seconds: float, count: int = 1) -> None:
        if seconds < 0:
            seconds = 0.0
        self._seconds[phase] = self._seconds.get(phase, 0.0) + seconds
        self._counts[phase] = self._counts.get(phase, 0) + count

    def merge_worker_timings(self, timings: Optional[Dict[str, float]]) -> None:
        if not timings:
            return
        for phase, seconds in timings.items():
            self.add(phase, seconds, count=1)

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(phase, time.perf_counter() - start)

    def total_seconds(self) -> float:
        return sum(self._seconds.values())

    def category_totals(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for phase, seconds in self._seconds.items():
            category = PHASE_CATEGORIES.get(phase, "other")
            totals[category] = totals.get(category, 0.0) + seconds
        return totals

    def report(self, title: str = "create_dataset timing") -> None:
        if not self._seconds:
            logger.info("No timing data recorded for %s", title)
            return ""

        total = self.total_seconds()
        rows = sorted(self._seconds.items(), key=lambda x: x[1], reverse=True)

        col_phase = max(len("Phase"), max(len(r[0]) for r in rows))
        col_time = len("Time")
        col_pct = len("% total")
        col_n = len("N")

        header = (
            f"{'Phase':<{col_phase}}  {'Time':>{col_time}}  "
            f"{'% total':>{col_pct}}  {'N':>{col_n}}  Category"
        )
        sep = "-" * len(header)

        lines = [f"\n=== {title} ===", header, sep]
        for phase, seconds in rows:
            pct = (seconds / total * 100) if total > 0 else 0.0
            category = PHASE_CATEGORIES.get(phase, "other")
            label = CATEGORY_LABELS.get(category, category)
            n = self._counts.get(phase, 1)
            lines.append(
                f"{phase:<{col_phase}}  {format_time(seconds):>{col_time}}  "
                f"{pct:>{col_pct}.1f}  {n:>{col_n}}  {label}"
            )

        lines.append(sep)
        lines.append(f"{'TOTAL (sum of phases)':<{col_phase}}  {format_time(total):>{col_time}}  100.0")

        cat_totals = self.category_totals()
        lines.append("")
        lines.append("=== Summary by category ===")
        cat_rows = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
        for category, seconds in cat_rows:
            label = CATEGORY_LABELS.get(category, category)
            pct = (seconds / total * 100) if total > 0 else 0.0
            lines.append(f"  {label:<28} {format_time(seconds):>12}  ({pct:.1f}%)")

        lines.append("")
        lines.extend(self._analysis_lines(cat_totals, total, rows))
        report_text = "\n".join(lines)
        logger.info(report_text)
        return report_text

    def _analysis_lines(
        self,
        cat_totals: Dict[str, float],
        total: float,
        rows: list,
    ) -> list[str]:
        if total <= 0:
            return ["Analysis: no measurable time."]

        index_t = cat_totals.get("index", 0.0)
        protein_t = cat_totals.get("protein_extraction", 0.0)
        save_t = cat_totals.get("file_save", 0.0)
        orch_t = cat_totals.get("orchestration", 0.0)

        parts = sorted(
            [
                ("index processing", index_t),
                ("protein extraction", protein_t),
                ("file saving", save_t),
                ("batch orchestration", orch_t),
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        dominant = parts[0]
        lines = [
            "Analysis:",
            f"  Dominant cost: {dominant[0]} ({dominant[1] / total * 100:.1f}% of tracked time).",
        ]
        if len(parts) > 1 and parts[1][1] > 0:
            lines.append(
                f"  Second: {parts[1][0]} ({parts[1][1] / total * 100:.1f}%)."
            )
        if index_t > 0 and protein_t > 0:
            ratio = protein_t / index_t
            lines.append(
                f"  Protein extraction vs index processing: {ratio:.1f}x "
                f"({'extraction dominates' if ratio > 1 else 'index work is relatively heavy'})."
            )
        slowest_phase = rows[0][0] if rows else None
        if slowest_phase:
            lines.append(f"  Slowest single phase: {slowest_phase}.")
        return lines


def measure_worker_pipeline(
    client,
    path_for_batch,
    pdb_ids,
    workers,
    *,
    map_read,
    map_convert,
    aggregate_fn,
    save_fn,
    extra_gather_tasks=None,
):
    """
    Run read -> convert -> aggregate -> H5 save with per-stage timings.

    Returns (primary_results..., worker_timings dict).
    """
    start = time.perf_counter()
    read_futures = client.map(map_read, pdb_ids, workers=workers)
    convert_futures = client.map(map_convert, read_futures, workers=workers)
    download_start = time.time()
    aggregated_future = client.submit(
        aggregate_fn, convert_futures, download_start, workers=workers
    )
    aggregated = client.gather(aggregated_future)
    pipeline_time = time.perf_counter() - start

    h5_start = time.perf_counter()
    h5_future = client.submit(
        save_fn, path_for_batch, aggregated, pure=False, workers=workers
    )
    extra_futures = extra_gather_tasks or []
    gathered = client.gather([h5_future, *extra_futures])
    h5_time = time.perf_counter() - h5_start

    timings = {
        "protein_extraction_pipeline": pipeline_time,
        "h5_save": h5_time,
    }
    return gathered, timings


_active_timings: Optional[CreateDatasetTimings] = None
_installed = False


def get_active_timings() -> Optional[CreateDatasetTimings]:
    return _active_timings


def install_create_dataset_timing() -> None:
    """Patch create_dataset pipeline to record and report phase timings."""
    global _installed
    if _installed:
        return
    _installed = True

    from toolbox.models.manage_dataset.collection_type import CollectionType
    from toolbox.models.manage_dataset.database_type import DatabaseType
    from toolbox.models.manage_dataset.index.handle_index import create_index, read_index
    from toolbox.models.manage_dataset import structures_dataset as sd
    from toolbox.models.manage_dataset import compute_batches as cb_mod
    from toolbox.utlis.logging import log_title

    orig_create = sd.StructuresDataset.create_dataset
    orig_download = sd.StructuresDataset.download_ids
    orig_compute = cb_mod.ComputeBatches.compute

    def timed_compute(self, inputs, factor=1):
        start = time.perf_counter()
        try:
            return orig_compute(self, inputs, factor=factor)
        finally:
            if _active_timings is not None:
                _active_timings.add("batch_orchestration", time.perf_counter() - start)

    cb_mod.ComputeBatches.compute = timed_compute

    def timed_download_ids(self, ids):
        timings = _active_timings
        if timings is None:
            return orig_download(self, ids)

        from toolbox.models.manage_dataset.extract_archive import (
            extract_archive,
            save_extracted_files,
        )

        if self.input_path is not None:
            with timings.measure("extract_archive"):
                extracted_path = extract_archive(self.input_path, self)
            if extracted_path is not None:
                missing_ids = save_extracted_files(self, extracted_path, ids)
                ids = missing_ids if missing_ids is not None else []
            else:
                return

        missing_files = None
        match self.db_type:
            case DatabaseType.PDB:
                missing_files = self.handle_pdb(ids)
            case DatabaseType.AFDB:
                missing_files = self.handle_afdb(ids)
            case DatabaseType.ESMatlas:
                missing_files = self.handle_esma()
            case DatabaseType.other:
                pass

        if not missing_files:
            return

        with timings.measure("write_missing_ids_file"):
            with open(self.dataset_path() / "missing_ids_files.txt", "w") as f:
                for file in missing_files:
                    f.write(file + "\n")
        logger.info(
            "Missing files saved to %s",
            self.dataset_path() / "missing_ids_files.txt",
        )

    def timed_create_dataset(self):
        global _active_timings
        log_title("Creating dataset")
        timings = CreateDatasetTimings()
        _active_timings = timings
        self._create_dataset_timings = timings

        try:
            with timings.measure("client_setup"):
                self.add_client()

            if self.collection_type == CollectionType.subset:
                if self.ids_file is None:
                    raise ValueError("Subset collection type requires ids_file")
                if not self.ids_file.exists():
                    raise FileNotFoundError(f"ids_file {self.ids_file} does not exist")

            self.dataset_repo_path().mkdir(exist_ok=True, parents=True)
            self.dataset_path().mkdir(exist_ok=True, parents=True)

            if (
                self.collection_type is CollectionType.subset
                or self.collection_type is CollectionType.all
            ):
                if self.overwrite:
                    with timings.measure("requested_ids"):
                        present_file_paths = {}
                        missing_ids = self.requested_ids()
                else:
                    with timings.measure("read_indexes"):
                        self._handle_indexes.read_indexes("dataset")
                    with timings.measure("requested_ids"):
                        requested_ids = self.requested_ids()
                    with timings.measure("find_present_and_missing"):
                        present_file_paths, missing_ids = (
                            self._handle_indexes.find_present_and_missing_ids(
                                "dataset", requested_ids
                            )
                        )

                with timings.measure("create_index_present"):
                    create_index(
                        self.dataset_index_file_path(),
                        present_file_paths,
                        self.config.data_path,
                    )

                if (
                    self.db_type == DatabaseType.other
                    and self.collection_type == CollectionType.subset
                    and len(missing_ids) > 0
                ):
                    raise RuntimeError(
                        "Missing ids are not allowed when subsetting all DBs!"
                    )

                if len(missing_ids) > 0:
                    with timings.measure("download_ids_total"):
                        timed_download_ids(self, missing_ids)
            else:
                with timings.measure("download_ids_total"):
                    timed_download_ids(self, None)

            with timings.measure("read_index_final"):
                index = read_index(
                    self.dataset_index_file_path(), self.config.data_path
                )

            if len(index.keys()) == 0:
                raise sd.FatalDatasetError("No files found in dataset")

            with timings.measure("save_dataset_metadata"):
                self.save_dataset_metadata()

            report = timings.report("create_dataset timing")
            if report:
                timing_path = self.dataset_path() / "create_dataset_timing.txt"
                try:
                    timing_path.write_text(report, encoding="utf-8")
                    logger.info("Timing report saved to %s", timing_path)
                except OSError as err:
                    logger.warning("Could not write timing report to %s: %s", timing_path, err)
        finally:
            _active_timings = None

    orig_download_pdb = sd.StructuresDataset._download_pdb_

    def timed_download_pdb(self, ids):
        timings = _active_timings
        if timings is None:
            return orig_download_pdb(self, ids)

        from toolbox.models.manage_dataset.utils import mkdir_for_batches, retrieve_pdb_chunk_to_h5
        from toolbox.models.manage_dataset.compute_batches import ComputeBatches
        from toolbox.models.utils.create_client import total_workers
        from pathlib import Path

        Path(self.structures_path()).mkdir(exist_ok=True, parents=True)
        pdb_repo_path = self.structures_path()
        batch_offset = self.batches_count()
        chunks = list(self.chunk(ids))

        with timings.measure("mkdir_for_batches"):
            mkdir_for_batches(pdb_repo_path, len(chunks), offset=batch_offset)

        logger.info(
            "Downloading %s PDBs into %s new chunks (offset by %s)",
            len(ids),
            len(chunks),
            batch_offset,
        )

        new_files_index = {}
        downloaded_structures_index: dict = {}

        def run(input_data, machine):
            return self._client.submit(
                retrieve_pdb_chunk_to_h5,
                *input_data,
                self.binary_data_download,
                [machine],
                workers=[machine],
            )

        def collect(result):
            if len(result) == 4:
                downloaded_pdbs, file_path, dl_map, worker_timings = result
                timings.merge_worker_timings(worker_timings)
            else:
                downloaded_pdbs, file_path, dl_map = result
            new_files_index.update({k: file_path for k in downloaded_pdbs})
            downloaded_structures_index.update(dl_map)

        compute_batches = ComputeBatches(
            self._client,
            run,
            collect,
            "pdb" + "_b" if self.binary_data_download else "",
            len(chunks),
        )

        inputs = (
            (pdb_repo_path / f"{i + batch_offset}", ids_chunk)
            for i, ids_chunk in enumerate(chunks)
        )

        factor = 10
        factor = 15 if total_workers() > 1500 else factor
        factor = 20 if total_workers() > 2000 else factor
        compute_batches.compute(inputs, factor=factor)

        logger.info("Adding new files to index")
        try:
            logger.info("Extracted %s new protein chain(s)", len(new_files_index))
            with timings.measure("update_dataset_index"):
                self.add_new_files_to_index(new_files_index)
            with timings.measure("update_downloaded_structures_index"):
                self.add_downloaded_structures_to_index(downloaded_structures_index)
        except Exception as e:
            logger.error("Failed to update index: %s", e)

    from toolbox.models.manage_dataset.utils import install_afdb_download_prefilter

    install_afdb_download_prefilter()

    timed_create_dataset._timing_installed = True  # type: ignore[attr-defined]
    sd.StructuresDataset.create_dataset = timed_create_dataset
    sd.StructuresDataset.download_ids = timed_download_ids
    sd.StructuresDataset._download_pdb_ = timed_download_pdb
