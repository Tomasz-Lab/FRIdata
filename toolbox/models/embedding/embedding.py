from datetime import datetime

from pathlib import Path
from typing import Dict, List, ClassVar

import dask.bag as db

from toolbox.models.embedding.embedder.embedder_type import EmbedderType
from toolbox.models.manage_dataset.index.handle_index import create_index, read_index
from toolbox.models.manage_dataset.index.handle_indexes import HandleIndexes, SearchIndexResult
from toolbox.models.manage_dataset.utils import format_time

import time

from toolbox.utlis.logging import log_title
from toolbox.utlis.logging import logger


class Embedding:
    embeddings_index_path: Path
    outputs_dir: Path
    structures_dataset: "StructuresDataset"

    Fasta_file: ClassVar[str] = "output.fasta"

    def __init__(self, structures_dataset: "StructuresDataset"):
        self.structures_dataset = structures_dataset
        # Initialize directories and paths using a helper function
        self._init_paths(
            self.structures_dataset.dataset_dir_name(),
            self.structures_dataset.embeddings_index_path()
        )

    def _init_paths(self, output_dir_name: str | None, embeddings_index_path: Path | None):
        """
        Helper method which initializes the output directories and embeddings index path.
        """
        self.embeddings_index_path = embeddings_index_path

        embeddings_path_obj = Path(self.structures_dataset.config.data_path) / "embeddings"
        if not embeddings_path_obj.exists():
            embeddings_path_obj.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M") if output_dir_name is None else output_dir_name
        embeddings_dir = embeddings_path_obj / timestamp
        embeddings_dir.mkdir(exist_ok=True, parents=True)

        self.outputs_dir = embeddings_dir


    def run(self):
        start = time.time()
        log_title("Getting embeddings")

        search_index_result = search_embedding_indexes(self.structures_dataset)

        missing_embeddings = search_index_result.missing_protein_files
        present_embeddings = search_index_result.present

        # Instead of opening the file and parsing here,
        # pass the missing IDs to missing_ids_to_fasta which returns a dict.

        sequences = self.missing_ids_to_fasta(missing_embeddings.keys())

        # Use the embedder type from the dataset, default to ESM2 if not specified
        if self.structures_dataset.embedder_type is None:
            self.structures_dataset.embedder_type = EmbedderType.ESM2_T33_650M
        if self.structures_dataset.embedding_size is None:
            self.structures_dataset.embedding_size = (
                self.structures_dataset.embedder_type.embedding_size
            )

        # Get the embedder instance and run embedding
        logger.info(f"Loading embedding model: {self.structures_dataset.embedder_type.value}")
        embedder = self.structures_dataset.embedder_type.create_embedder()
        index_of_new_embeddings = embedder.embed(sequences, self.outputs_dir)

        present_embeddings.update(index_of_new_embeddings)

        create_index(self.embeddings_index_path, present_embeddings, self.structures_dataset.config.data_path)

        self.structures_dataset.save_dataset_metadata()

        end = time.time()
        logger.info(f"Total time: {format_time(end - start)}\n")


    def missing_ids_to_fasta(self, missing_ids: List[str]) -> Dict[str, str]:
        """
        Process datasets of protein sequences to extract sequences for the missing IDs 
        and return a dictionary mapping protein IDs to their sequences.
        
        :param missing_ids: List of protein IDs that need to be processed.
        :return: Dictionary of protein_id -> sequence.
        """
        start_time = time.time()

        index = read_index(self.structures_dataset.sequences_index_path(), self.structures_dataset.config.data_path)
        all_sequence_files = set(index.values())

        if not all_sequence_files:
            logger.warning("No sequence files found.")
            return {}

        # Read and merge all FASTA contents from the sequence files using Dask
        fasta_bag = db.from_sequence(all_sequence_files, partition_size=1)
        merged_fasta_list = fasta_bag.map(lambda file: Path(file).read_text()).compute()
        merged_fasta = "".join(merged_fasta_list)

        sequences = {}
        current_id = None
        current_seq = []

        # Parse merged FASTA content and build a dictionary for only those sequences 
        # whose header ID is in missing_ids.
        for line in merged_fasta.splitlines():
            line = line.strip()
            if line.startswith('>'):
                # If there is an ongoing sequence, store it
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                candidate_id = line[1:].split()[0]
                if candidate_id in missing_ids:
                    current_id = candidate_id
                    current_seq = []
                else:
                    current_id = None  # skip if not in missing_ids
            else:
                if current_id is not None:
                    current_seq.append(line)
        if current_id is not None and current_seq:
            sequences[current_id] = ''.join(current_seq)

        end_time = time.time()
        logger.info(f"Extracting sequences for embeddings took: {format_time(end_time - start_time)}")
        return sequences


def search_embedding_indexes(
    structures_dataset: "StructuresDataset",
) -> SearchIndexResult:
    protein_index = read_index(structures_dataset.dataset_index_file_path(), structures_dataset.config.data_path)

    handle_indexes: HandleIndexes = structures_dataset._handle_indexes

    search_index_result = handle_indexes.full_handle("embeddings", protein_index, structures_dataset.overwrite)

    return search_index_result
