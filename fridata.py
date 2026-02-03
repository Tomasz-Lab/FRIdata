import argparse
import pathlib
import sys


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


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        "-d",
        "--db",
        required=True,
        choices=db_types,
        metavar="name",
        help=f"Database Types: {' '.join(db_types)}",
    )
    parser.add_argument(
        "-c",
        "--collection",
        required=True,
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
        help="Inputs to generate",
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

    parser_dataset = subparsers.add_parser("dataset", help="Create protein dataset")
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

    load_dataset_parser = subparsers.add_parser("load", help="Load a dataset from json")
    add_common_arguments(load_dataset_parser)

    extract_sequence_and_coordinates_parser = subparsers.add_parser(
        "generate_sequence", help="Generate sequences for "
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
        "generate_distograms", help="Generate distograms for "
    )
    add_common_arguments(generate_distograms_parser)

    read_distograms_parser = subparsers.add_parser(
        "read_distograms", help="Read distograms for "
    )
    add_common_arguments(read_distograms_parser)

    read_pdbs_parser = subparsers.add_parser(
        "read_pdbs", help="Read pdbs for "
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
        "verify_chains", help="Verify chains for "
    )
    add_common_arguments(verify_chains_parser)

    create_archive_parser = subparsers.add_parser(
        "create_archive", help="Create PDB compressed archive"
    )
    add_common_arguments(create_archive_parser)

    input_generation_parser = subparsers.add_parser(
        "input_generation", help="Create dataset, generate sequences distograms and embeddings"
    )
    input_generation_parser.add_argument(
        "--slurm", action="store_true", help="Use SLURM job scheduler"
    )
    add_dataset_parser_arguments(input_generation_parser)
    add_embedder_argument(input_generation_parser, required=True)

    # export_index_view command
    export_index_parser = subparsers.add_parser(
        "export_index_view", help="Export single-file HTML report for dataset indexes"
    )
    export_index_parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=None,
        help="Path to dataset directory or its dataset.json",
    )
    export_index_parser.add_argument(
        "--dataset-slug",
        type=str,
        default=None,
        help="Dataset slug (folder suffix after --)",
    )
    export_index_parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=None,
        help="Override datasets root (default: <data_path>/datasets)",
    )
    export_index_parser.add_argument(
        "--index-types",
        type=str,
        default="all",
        help="Comma-separated index types (dataset,sequences,coordinates,embeddings,distograms) or 'all'",
    )
    export_index_parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for reports (default: <repo_root>/reports)",
    )



    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
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
