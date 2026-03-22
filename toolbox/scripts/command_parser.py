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

    def dataset(self):
        from toolbox.models.embedding.embedder.embedder_type import EmbedderType
        from toolbox.models.manage_dataset.structures_dataset import StructuresDataset
        from toolbox.models.manage_dataset.utils import format_time

        start = time.time()

        # Convert embedder string to EmbedderType enum if provided
        embedder_type = None
        if hasattr(self.args, "embedder") and self.args.embedder:
            for embedder_enum in EmbedderType:
                if embedder_enum.value == self.args.embedder:
                    embedder_type = embedder_enum
                    break

        dataset = StructuresDataset(
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
        create_archive(self.structures_dataset)

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
            dataset=getattr(self.args, "dataset", None),
            dataset_slug=getattr(self.args, "dataset_slug", None),
            root=getattr(self.args, "root", None),
            index_types=index_types,
            output_dir=getattr(self.args, "output_dir", None),
        )
        logger.info(f"Report generated: {out_path}")

    def input_generation(self):
        from toolbox.models.manage_dataset.structures_dataset import FatalDatasetError
        from toolbox.models.manage_dataset.utils import format_time

        total_time = time.time()
        is_error = False
        try:
            self.dataset()
        except FatalDatasetError as e:
            logger.error("Fatal error! Exiting...")
            logger.error(e)
            return
        except Exception as e:
            print_exc(e)
            is_error = True
        try:
            self.structures_dataset.extract_sequence_and_coordinates()
        except Exception as e:
            print_exc(e)
            is_error = True
        try:
            self.structures_dataset.generate_distograms()
        except Exception as e:
            print_exc(e)
            is_error = True
        try:
            self.structures_dataset.generate_embeddings()
        except Exception as e:
            print_exc(e)
            is_error = True
        end_time = time.time()
        logger.info(f"Total time for all steps: {format_time(end_time - total_time)}")
        if is_error:
            logger.info("Error! Exiting...")
        else:
            logger.info("\nComputation successfully completed!")

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
