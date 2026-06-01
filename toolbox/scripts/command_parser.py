from __future__ import annotations

from argparse import Namespace
import json
import logging
import sys
import traceback
import time
from pathlib import Path
from typing import TYPE_CHECKING

from toolbox.config import Config
from toolbox.models.manage_dataset.index.handle_index import read_index
from toolbox.utlis.logging import logger

if TYPE_CHECKING:
    from toolbox.models.manage_dataset.structures_dataset import StructuresDataset


class CommandParser:
    def __init__(self, args: Namespace, config: Config):
        self.structures_dataset = None
        self.args = args
        self.config = config

    def _create_dataset_from_path_(self) -> StructuresDataset:
        from toolbox.models.manage_dataset.structures_dataset import StructuresDataset
        from toolbox.models.utils.create_client import create_client

        if self.structures_dataset is not None:
            return self.structures_dataset
        path = self.args.file_path
        if path.is_dir() and (path / "dataset.json").exists():
            self.structures_dataset = StructuresDataset.model_validate_json(
                (path / "dataset.json").read_text()
            )
        elif path.is_file():
            if path.suffix != ".json":
                logger.error("Dataset path is not valid")
                raise FileNotFoundError
            self.structures_dataset = StructuresDataset.model_validate_json(
                path.read_text()
            )
        else:
            logger.error("Dataset path is not valid")
            raise FileNotFoundError

        self.structures_dataset._client = create_client(
            True if self.args.slurm else self.structures_dataset.is_hpc_cluster
        )
        return self.structures_dataset

    def _log_command(self):
        """Log the complete command line that started the program."""
        full_command = " ".join(sys.argv)
        logger.info(f"Started with command: {full_command}")

    def _configure_dataset_logging(self):
        """Configure logging to dataset log file if not already specified."""
        if not hasattr(self.args, "log_file") or self.args.log_file is None:
            from toolbox.utlis.colored_logging import setup_logging_with_file

            log_level = logging.DEBUG if self.args.verbose else logging.INFO
            log_format = "%(asctime)s %(levelname)s %(message)s"
            setup_logging_with_file(
                level=log_level,
                fmt=log_format,
                log_file=self.structures_dataset.log_file_path(),
            )
            logger.info(f"Logging configured to: {self.structures_dataset.log_file_path()}")
            # Log the complete command line for dataset operations
            self._log_command()


    def _build_structures_dataset_from_args(self) -> StructuresDataset:
        """Build StructuresDataset from CLI args. Used by create_dataset and generate_data."""
        from toolbox.models.embedding.embedder.embedder_type import EmbedderType
        from toolbox.models.manage_dataset.structures_dataset import StructuresDataset
        
        embedder_type = None
        if hasattr(self.args, "embedder") and self.args.embedder:
            for embedder_enum in EmbedderType:
                if embedder_enum.value == self.args.embedder:
                    embedder_type = embedder_enum
                    break

        return StructuresDataset(
            db_type=self.args.db,
            collection_type=self.args.collection,
            proteome=self.args.proteome,
            version=self.args.version,
            ids_file=self.args.ids,
            seqres_file=self.args.seqres,
            archive_path=self.args.archive,
            overwrite=self.args.overwrite,
            batch_size=(
                None if self.args.batch_size is None else int(self.args.batch_size)
            ),
            binary_data_download=self.args.binary,
            is_hpc_cluster=self.args.slurm,
            input_path=self.args.input_path,
            verbose=self.args.verbose if hasattr(self.args, "verbose") else False,
            config=self.config,
            embedder_type=embedder_type,
        )

    def create_dataset(self):
        from toolbox.models.manage_dataset.utils import format_time
        
        start = time.time()

        dataset = self._build_structures_dataset_from_args()
        self.structures_dataset = dataset

        # Configure logging to dataset log file if not already specified
        self._configure_dataset_logging()

        dataset.create_dataset()

        # Print dataset name in special format for shell script parsing
        dataset_name = dataset.dataset_dir_name()
        print(f"DATASET_NAME:{dataset_name}")

        end = time.time()
        logger.info(f"Total time: {format_time(end - start)}")
        return dataset

    def generate_embeddings(self):
        from toolbox.models.embedding.embedder.embedder_type import EmbedderType

        self._create_dataset_from_path_()

        # Configure logging to dataset log file if not already specified
        self._configure_dataset_logging()

        # Set embedder type if provided
        if hasattr(self.args, "embedder") and self.args.embedder:
            for embedder_enum in EmbedderType:
                if embedder_enum.value == self.args.embedder:
                    self.structures_dataset.embedder_type = embedder_enum
                    break

        self.structures_dataset.generate_embeddings()

    def load(self):
        dataset = self._create_dataset_from_path_()
        logger.info(dataset)

    def generate_sequence(self):
        self._create_dataset_from_path_()

        # Configure logging to dataset log file if not already specified
        self._configure_dataset_logging()

        self.structures_dataset.extract_sequence_and_coordinates(
            self.args.ca_mask, self.args.no_substitution
        )

    def generate_distograms(self):
        self._create_dataset_from_path_()

        # Configure logging to dataset log file if not already specified
        self._configure_dataset_logging()

        self.structures_dataset.generate_distograms()

    def read_distograms(self):
        from toolbox.models.manage_dataset.distograms.generate_distograms import (
            read_distograms_from_file,
        )

        logger.info(read_distograms_from_file(self.args.file_path))

    def read_pdbs(self):
        read_pdbs(
            self.args.file_path, self.args.ids, self.args.to_directory, self.args.print
        )

    def verify_chains(self):
        from toolbox.models.chains.verify_chains import verify_chains

        self._create_dataset_from_path_()
        verify_chains(self.structures_dataset, "./toolbox/pdb_seqres.txt")

    def create_archive(self):
        from toolbox.scripts.archive import create_archive

        self._create_dataset_from_path_()
        create_archive(
            self.structures_dataset,
            keep_shard_zips=getattr(self.args, "keep_shard_zips", False),
            archive_types=getattr(self.args, "type", None),
        )

    def inspect_h5(self):
        from toolbox.utlis.inspect_h5 import inspect_h5

        inspect_h5(
            Path(self.args.file),
            mode=getattr(self.args, "mode", "structure"),
            names=getattr(self.args, "pdb_names", None),
        )

    def inspect_idx(self):
        from toolbox.utlis.inspect_idx import inspect_idx

        inspect_idx(Path(self.args.file))

    def remove_dataset(self):
        from toolbox.utlis.remove_dataset import remove_dataset

        remove_dataset(self.args.name, self.config)

    def create_dashboard(self):
        # CLI handler for create_dashboard (formerly export_index_view)
        # Uses global config from args/config loaded in fridata.py
        from toolbox.viewer.export_index_html import export_index_view

        index_types = None
        if (
            hasattr(self.args, "index_types")
            and self.args.index_types
            and self.args.index_types != "all"
        ):
            index_types = [
                s.strip()
                for s in self.args.index_types.split(",")
                if s.strip()
            ]

        out_path = export_index_view(
            config=self.config,
            dataset_ref=getattr(self.args, "dataset", None),
            root=getattr(self.args, "root", None),
            index_types=index_types,
            output_dir=getattr(self.args, "output_dir", None),
        )
        logger.info(f"Report generated: {out_path}")

    def _finish_generate_data(self, started_at: float, is_error: bool) -> None:
        from toolbox.models.manage_dataset.utils import format_time

        logger.info(f"Total time for all steps: {format_time(time.time() - started_at)}")
        if is_error:
            logger.error("Error! Exiting...")
        else:
            logger.info("Computation successfully completed!")

    def generate_data(self):
        from toolbox.models.manage_dataset.structures_dataset import FatalDatasetError

        total_time = time.time()
        is_error = False

        # Parse -t/--type: empty, "all" -> run all steps; otherwise run only specified
        type_arg = getattr(self.args, 'type', None) or ""
        type_arg = (type_arg or "").strip().lower()
        run_all = type_arg in ("", "all")
        if run_all:
            selected = {"dataset", "sequences", "coordinates", "distograms", "embeddings"}
        else:
            selected = {s.strip().lower() for s in type_arg.split(",") if s.strip()}
            valid = {"dataset", "sequences", "coordinates", "distograms", "embeddings"}
            invalid = selected - valid
            if invalid:
                logger.warning(f"Ignoring unknown -t values: {invalid}")

        # Step 1: dataset
        run_dataset = "dataset" in selected or run_all
        try:
            if run_dataset:
                self.create_dataset()
            else:
                self.structures_dataset = self._build_structures_dataset_from_args()
                self.structures_dataset.add_client()
                self._configure_dataset_logging()

        except FatalDatasetError as e:
            logger.error("Fatal error! Exiting...")
            logger.error(e)
            self._finish_generate_data(total_time, is_error=True)
            return
        except Exception as e:
            print_exc(e)
            is_error = True

        ds = self.structures_dataset
        needs_structure_index = (
            "sequences" in selected
            or "coordinates" in selected
            or "distograms" in selected
            or "embeddings" in selected
            or run_all
        )
        if ds is None:
            if needs_structure_index:
                logger.error(
                    "Dataset setup did not complete; cannot run sequences, coordinates, "
                    "distograms, or embeddings. See earlier traceback if an exception was logged."
                )
            self._finish_generate_data(total_time, is_error=True)
            return

        if needs_structure_index and not run_dataset:
            idx_path = ds.dataset_index_file_path()
            if not idx_path.exists():
                logger.error(
                    "Dataset index not found: %s. Run generate_data with 'dataset' in -t "
                    "(e.g. -t dataset,sequences,...) or run create_dataset first so the structure index is created.",
                    idx_path.resolve(),
                )
                self._finish_generate_data(total_time, is_error=True)
                return

        # Step 2: sequences and coordinates (single step produces both)
        if "sequences" in selected or "coordinates" in selected or run_all:
            sequences_and_coordinates_ok = False
            try:
                ds.extract_sequence_and_coordinates()
                sequences_and_coordinates_ok = True
            except Exception as e:
                print_exc(e)
                is_error = True
            if sequences_and_coordinates_ok:
                seq_idx = read_index(
                    ds.sequences_index_path(),
                    ds.config.data_path,
                )
                if len(seq_idx) == 0:
                    logger.error(
                        "No proteins to process after sequences/coordinates step; skipping distograms and embeddings."
                    )
                    self._finish_generate_data(total_time, is_error=True)
                    return

        # Step 3: distograms
        if "distograms" in selected or run_all:
            try:
                ds.generate_distograms()
            except Exception as e:
                print_exc(e)
                is_error = True

        # Step 4: embeddings
        if "embeddings" in selected or run_all:
            try:
                ds.generate_embeddings()
            except Exception as e:
                print_exc(e)
                is_error = True

        self._finish_generate_data(total_time, is_error)

    def run(self):
        command_method = getattr(self, self.args.command)
        if command_method:
            command_method()
            self.cleanup()
        else:
            raise ValueError(f"Unknown command - {self.args.command}")

    def cleanup(self):
        ds = getattr(self, "structures_dataset", None)
        client = getattr(ds, "_client", None) if ds is not None else None
        if client:
            import warnings

            import distributed

            warnings.simplefilter("ignore", distributed.comm.core.CommClosedError)

            # Suppress noisy tornado/asyncio tracebacks that fire during
            # nanny shutdown (TimeoutError / CancelledError).  These are
            # harmless – the work is already done – but alarming for users.
            logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
            logging.getLogger("distributed.nanny").setLevel(logging.CRITICAL)
            logging.getLogger("distributed.process").setLevel(logging.CRITICAL)

            cluster = getattr(client, "cluster", None)
            try:
                client.close()
            except Exception:
                pass
            if cluster is not None:
                try:
                    cluster.close()
                except Exception:
                    pass
            ds._client = None


def print_exc(e):
    logger.error(f"Error ({type(e)}): {str(e)}")
    logger.error(traceback.format_exc())


def read_pdbs(file_path, ids, to_directory, is_print):
    from toolbox.models.manage_dataset.utils import read_pdbs_from_h5

    if ids.exists():
        ids = ids.read_text().splitlines()

    pdbs_dict = read_pdbs_from_h5(file_path, ids)

    if is_print:
        logger.info(json.dumps(pdbs_dict))

    if to_directory:
        extract_dir: Path = to_directory
        if not extract_dir.exists() and not extract_dir.is_dir():
            logger.error("ERROR: Provided output path doesn't exist")
            return
    else:
        extract_dir = file_path.parent / f"extracted_{file_path.stem}"
        extract_dir.mkdir(exist_ok=True, parents=True)

    def save_pdb(pdb_code, pdb_file):
        file_name = f"{pdb_code}" if pdb_code.endswith(".pdb") else f"{pdb_code}.pdb"
        logger.info(f"Saving {file_name}")
        with open(extract_dir / file_name, "w") as f:
            f.write(pdb_file)

    logger.info("Extracting PDB files")
    for pdb_code, pdb_file in pdbs_dict.items():
        save_pdb(pdb_code, pdb_file)
    logger.info("Extraction complete")
