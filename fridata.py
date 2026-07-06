import argparse
import pathlib
import sys


from toolbox.models.manage_dataset.utils import install_afdb_download_prefilter

install_afdb_download_prefilter()

from toolbox.models.manage_dataset.database_type import DatabaseType
from toolbox.models.manage_dataset.collection_type import CollectionType
from toolbox.models.embedding.embedder.embedder_type import EmbedderType
from toolbox.scripts.command_parser import CommandParser

db_types = DatabaseType._member_names_
collection_types = CollectionType._member_names_
embedder_types = [member.value for member in EmbedderType]

import logging
from toolbox.utlis.logging import logger, setup_colored_logging
from toolbox.utlis.colored_logging import setup_logging_with_file

def add_common_arguments(parser):
    parser.add_argument("--slurm", action="store_true", help="Use SLURM job scheduler")
    parser.add_argument("-p", "--file-path", required=True, type=pathlib.Path)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-file", type=pathlib.Path, help="Path to log file for file logging")


def add_dataset_parser_arguments(parser, require_db_collection=True):
    parser.add_argument(
        "-d",
        "--db",
        required=require_db_collection,
        choices=db_types,
        metavar="name",
        help=f"Database Types: {' '.join(db_types)}",
    )
    parser.add_argument(
        "-c",
        "--collection",
        required=require_db_collection,
        choices=collection_types,
        metavar="name",
        help=f"Collection Types: {' '.join(collection_types)}",
    )
    parser.add_argument(
        "-t",
        "--type",
        required=False,
        default="",
        metavar="name",
        help="Comma-separated types to generate: dataset, sequences, coordinates, distograms, embeddings; or 'all'. For generate_data only; ignored by create_dataset.",
        nargs="?",
    )
    parser.add_argument(
        "--proteome",
        required=False,
        default="",
        metavar="name",
        help="Precise proteome of AFDB part dataset",
        nargs="?",
    )
    parser.add_argument(
        "--version",
        required=False,
        help="String to differentiate datasets; default: current date",
    )
    parser.add_argument(
        "-i",
        "--ids",
        required=False,
        type=pathlib.Path,
        help="File with ids to create subset",
    )
    parser.add_argument(
        "-s",
        "--seqres",
        required=False,
        type=pathlib.Path,
        help="fasta file to use as sequence source",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Should overwrite existing files? Default - false",
    )
    parser.add_argument("-b", "--batch-size", type=str, default=None)
    parser.add_argument(
        "--binary", action="store_true", help="Download binary CIF in PDB db"
    )
    parser.add_argument(
        "--input-path",
        type=pathlib.Path,
        default=None,
        help="Path to input directory or archive (zip/tar.gz) with protein files (pdb/cif)",
    )
    parser.add_argument(
        "--archive",
          type=pathlib.Path,
            help='Path to tar.gz archive containing structure files'
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging mode"
    )


def add_embedder_argument(parser, required=True):
    parser.add_argument(
        "-e",
        "--embedder",
        required=required,
        choices=embedder_types,
        metavar="name",
        help=f"Embedder Types: {' '.join(embedder_types)}",
    )


def _parse_comma_separated_pdb_names(value: str) -> list[str]:
    names = [s.strip() for s in value.split(",") if s.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--name requires at least one non-empty PDB key")
    return names


def configure_logging(verbose, log_file=None):
    """Configure logging based on verbose flag and optional log file"""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s %(levelname)s %(message)s'
    
    # Set up logging with file support if log_file is provided
    if log_file:
        setup_logging_with_file(level=log_level, fmt=log_format, log_file=log_file)
    else:
        # Set up colored logging with the specified level and format
        setup_colored_logging(level=log_level, fmt=log_format)
    
    # When verbose is false, filter out logs with (V) prefix unless they're errors
    if not verbose:
        class VerboseFilter(logging.Filter):
            def filter(self, record):
                # Still show ERROR or higher regardless of (V) tag
                if record.levelno >= logging.ERROR:
                    return True
                # Filter out messages with (V) prefix in non-verbose mode
                return "(V)" not in record.getMessage()
                
        logger.addFilter(VerboseFilter())




def validate_inspect_h5_cli_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject invalid ``inspect_h5`` flag combinations after ``parse_args``."""
    if getattr(args, "command", None) != "inspect_h5":
        return
    if getattr(args, "pdb_names", None) and getattr(args, "mode", None) not in (
        "structures",
        "distograms",
        "coordinates",
        "embeddings",
    ):
        parser.error(
            "--name is only valid with --mode structures|distograms|coordinates|embeddings"
        )


def validate_generate_data_cli_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject invalid ``generate_data`` flag combinations after ``parse_args``."""
    if getattr(args, "command", None) != "generate_data":
        return
    has_path = getattr(args, "file_path", None) is not None
    if not has_path:
        if not args.db or not args.collection:
            parser.error(
                "generate_data requires -d/--db and -c/--collection, or -p/--file-path"
            )
        if not args.embedder:
            parser.error(
                "generate_data requires -e/--embedder when creating a new dataset (no -p)"
            )


def create_parser():
    parser = argparse.ArgumentParser(description="Create protein dataset")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-file", type=pathlib.Path, help="Path to log file for file logging")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help="Path to config JSON file (default: ./config.json in main directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    parser_dataset = subparsers.add_parser("create_dataset", help="Create protein dataset")
    parser_dataset.add_argument(
        "--slurm", action="store_true", help="Use SLURM job scheduler"
    )
    add_dataset_parser_arguments(parser_dataset)
    add_embedder_argument(parser_dataset, required=False)

    embedding_parser = subparsers.add_parser(
        "generate_embeddings", help="Create embeddings from datasets"
    )
    add_common_arguments(embedding_parser)
    add_embedder_argument(embedding_parser, required=True)


    extract_sequence_and_coordinates_parser = subparsers.add_parser(
        "generate_sequence", help="Generate sequences for..."
    )
    add_common_arguments(extract_sequence_and_coordinates_parser)
    extract_sequence_and_coordinates_parser.add_argument(
        "--ca_mask",
        action="store_true",
        help="Require a carbon alpha atom to include an amino acid in a sequence",
    )
    extract_sequence_and_coordinates_parser.add_argument(
        "--no_substitution",
        action="store_false",
        help="Don't substitute non standard amino acids",
    )

    generate_distograms_parser = subparsers.add_parser(
        "generate_distograms", help="Generate distograms for..."
    )
    add_common_arguments(generate_distograms_parser)

    read_distograms_parser = subparsers.add_parser(
        "read_distograms", help="Read distograms for..."
    )
    add_common_arguments(read_distograms_parser)

    read_pdbs_parser = subparsers.add_parser(
        "read_pdbs", help="Read pdbs for..."
    )
    read_pdbs_parser.add_argument(
        "--print", action="store_true", help="Print PDB files to the terminal"
    )
    read_pdbs_parser.add_argument(
        "--to_directory", type=pathlib.Path, help="Extract PDB files to the provided directory"
    )
    read_pdbs_parser.add_argument(
        "-i",
        "--ids",
        required=False,
        type=pathlib.Path,
        help="File with ids to extract",
    )
    add_common_arguments(read_pdbs_parser)

    verify_chains_parser = subparsers.add_parser(
        "verify_chains", help="Verify chains for..."
    )
    add_common_arguments(verify_chains_parser)

    create_archive_parser = subparsers.add_parser(
        "create_archive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help=(
            "Build one merged PDB .zip (archive_pdb_<dataset>_<timestamp>.zip) under "
            "data_path/archives/<dataset>/. Only proteins listed in the dataset index "
            "(per shard in dataset_reversed.idx) are included."
        ),
        epilog=(
            "Working directory: a `mate` helper (if you use it) often cd's into "
            "deepFRI2-toolbox-dev; after that, pass an absolute -p/--file-path to the "
            "dataset directory (the folder containing dataset.json), or cd to "
            "<data_path>/datasets first if you want to use a short folder name.\n"
            "Invocation: run `python fridata.py create_archive ...` from the repo "
            "unless fridata.py is installed on PATH."
        ),
    )
    create_archive_parser.add_argument(
        "--keep-shard-zips",
        action="store_true",
        help=(
            "Keep per-shard zip files under _staging/ after merge (default: remove staging)"
        ),
    )
    add_common_arguments(create_archive_parser)

    inspect_h5_parser = subparsers.add_parser(
        "inspect_h5", help="Read h5 file and display in vi editor"
    )
    inspect_h5_parser.add_argument(
        "file",
        type=pathlib.Path,
        help="Path to h5 file",
    )
    inspect_h5_parser.add_argument(
        "--mode",
        choices=[
            "preview",
            "structures",
            "keys",
            "distograms",
            "coordinates",
            "embeddings",
        ],
        default="preview",
        help=(
            "preview: H5 tree; keys: id list; structures: full PDB text (files group); "
            "distograms|coordinates|embeddings: numeric preview (or full dump with --name)"
        ),
    )
    inspect_h5_parser.add_argument(
        "--name",
        dest="pdb_names",
        type=_parse_comma_separated_pdb_names,
        default=None,
        metavar="KEYS",
        help=(
            "Comma-separated protein keys. Only with --mode "
            "structures|distograms|coordinates|embeddings. "
            "For numeric modes, emits a full array dump. "
            "Use inspect_h5 --mode keys to list keys."
        ),
    )
    inspect_h5_parser.add_argument(
        "--save",
        nargs="?",
        const="",
        default=None,
        metavar="FILEPATH",
        help=(
            "Write output to a file instead of vi. "
            "Omit FILEPATH to write a persistent temp file and print its path."
        ),
    )

    inspect_idx_parser = subparsers.add_parser(
        "inspect_idx", help="Read idx file (JSON), format as pretty JSON, display in vi"
    )
    inspect_idx_parser.add_argument(
        "file",
        type=pathlib.Path,
        help="Path to idx file",
    )

    remove_dataset_parser = subparsers.add_parser(
        "remove_dataset", help="Remove all traces of a dataset"
    )
    remove_dataset_parser.add_argument(
        "name",
        type=str,
        help="Dataset name (e.g. AFDB-subset--20250609_1333)",
    )

    generate_data_parser = subparsers.add_parser(
        "generate_data", help="Create dataset, generate sequences distograms and embeddings"
    )
    generate_data_parser.add_argument(
        "--slurm", action="store_true", help="Use SLURM job scheduler"
    )
    generate_data_parser.add_argument(
        "-p",
        "--file-path",
        type=pathlib.Path,
        default=None,
        help="Existing dataset directory (with dataset.json) or path to dataset.json",
    )
    add_dataset_parser_arguments(generate_data_parser, require_db_collection=False)
    add_embedder_argument(generate_data_parser, required=False)

    # create_dashboard command (formerly export_index_view)
    create_dashboard_parser = subparsers.add_parser(
        "create_dashboard", help="Export single-file HTML report for dataset indexes"
    )
    create_dashboard_parser.add_argument(
        "dataset",
        help=(
            "Dataset: filesystem path (directory or dataset.json), slug (folder suffix after --), "
            "or full folder name under --root when not an existing path"
        ),
    )
    create_dashboard_parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=None,
        help="Override datasets root (default: <data_path>/datasets)",
    )
    create_dashboard_parser.add_argument(
        "--index-types",
        type=str,
        default="all",
        help="Comma-separated index types (dataset,sequences,coordinates,embeddings,distograms) or 'all'",
    )
    create_dashboard_parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for reports (default: <repo_root>/reports)",
    )



    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    validate_inspect_h5_cli_args(args, parser)
    validate_generate_data_cli_args(args, parser)

    # Load config and raise if not found
    from toolbox.config import load_config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        raise e

    # Configure logging based on verbose flag and log file
    configure_logging(args.verbose, args.log_file)
    
    # Log the complete command line used to start the program
    full_command = " ".join(sys.argv)
    logger.info(f"Started with command: {full_command}")
    
    CommandParser(args, config).run()


if __name__ == "__main__":
    main()
