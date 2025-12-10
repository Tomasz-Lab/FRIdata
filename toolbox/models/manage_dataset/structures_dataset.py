import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any
from typing_extensions import Self

import dask.bag as db
import requests
from Bio.PDB import PDBList
from dask.distributed import Client, as_completed
from pydantic import BaseModel, model_validator
import requests
import tarfile
import shutil
from math import ceil

from toolbox.models.manage_dataset.collection_type import CollectionType
from toolbox.models.manage_dataset.compute_batches import ComputeBatches
from toolbox.models.manage_dataset.database_type import DatabaseType
from toolbox.models.manage_dataset.extract_archive import (
    extract_archive,
    save_extracted_files,
)
from toolbox.models.manage_dataset.index.handle_index import (
    create_index,
    add_new_files_to_index,
    read_index,
)
from toolbox.models.manage_dataset.index.handle_indexes import HandleIndexes
from toolbox.models.manage_dataset.sequences.sequence_and_coordinates_retriever import SequenceAndCoordinatesRetriever
from toolbox.models.manage_dataset.utils import (
    foldcomp_download,
    mkdir_for_batches,
    retrieve_pdb_chunk_to_h5,
    retrieve_afdb_chunk_to_h5,
    alphafold_chunk_to_h5,
    format_time,
    chunk
)
from toolbox.models.manage_dataset.distograms.generate_distograms import generate_distograms
from toolbox.models.embedding.embedding import Embedding
from toolbox.models.utils.from_archive import extract_batch_from_archive
from toolbox.models.utils.create_client import create_client, total_workers
from toolbox.utlis.filter_pdb_codes import filter_pdb_codes
from toolbox.utlis.logging import log_title
from toolbox.utlis.logging import logger
from toolbox.config import Config
from toolbox.models.embedding.embedder.embedder_type import EmbedderType

class FatalDatasetError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class StructuresDataset(BaseModel):
    db_type: DatabaseType
    collection_type: CollectionType
    proteome: str = ""
    version: Optional[str] = None
    ids_file: Optional[Path] = None
    seqres_file: Optional[Path] = None
    archive_path: Optional[Path] = None
    overwrite: bool = False
    batch_size: Optional[int] = None
    binary_data_download: bool = False
    is_hpc_cluster: bool = False
    input_path: Optional[Path] = None
    embedder_type: Optional[EmbedderType] = None
    embedding_size: Optional[int] = None
    _client: Optional[Client] = None
    _handle_indexes: Optional[HandleIndexes] = None
    _sequence_retriever: Optional[SequenceAndCoordinatesRetriever] = None
    created_at: Optional[str] = None
    config: Optional[Config] = None

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.version is None:
            self.version = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Automatically set embedding_size from embedder_type if not provided
        if self.embedder_type is not None and self.embedding_size is None:
            self.embedding_size = self.embedder_type.embedding_size
        if self.batch_size is None:
            self.batch_size = 1000
        if self.created_at is None:
            self.created_at = str(int(datetime.now().timestamp() * 1000000))
        return self

    def __init__(self, **data: Any):
        config = data.pop('config', None)
        super().__init__(**data)
        if isinstance(config, dict):
            self.config = Config(**config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ValueError("Invalid config")
        self._handle_indexes = HandleIndexes(self)
        self._sequence_retriever = SequenceAndCoordinatesRetriever(self) # SequenceRetriever(self)

    def dataset_repo_path(self):
        base_path = Path(self.config.data_path) if self.config else Path("/data")
        return (
            base_path / "structures"
            / self.db_type.name
            / f"{self.collection_type.name}_{self.proteome}"
            / self.version
        )

    def dataset_path(self):
        base_path = Path(self.config.data_path) if self.config else Path("/data")
        return base_path / "datasets" / self.dataset_dir_name()

    def log_file_path(self):
        """Get the path for the log file in the dataset directory."""
        return self.dataset_path() / "log.txt"

    def structures_path(self):
        return self.dataset_repo_path()

    def dataset_dir_name(self):
        sep = self.config.separator if self.config else "-"
        return f"{self.db_type.name}{sep}{self.collection_type.name}{sep}{self.proteome}{sep}{self.version}"

    def dataset_index_file_path(self) -> Path:
        return self.dataset_path() / "dataset.idx"

    def sequences_index_path(self):
        return self.dataset_path() / "sequences.idx"

    def distograms_index_path(self):
        return self.dataset_path() / "distograms.idx"

    def distograms_file_path(self):
        return self.dataset_path() / "distograms.h5"

    def embeddings_index_path(self):
        return self.dataset_path() / "embeddings.idx"

    def coordinates_index_path(self):
        return self.dataset_path() / "coordinates.idx"
    
    def input_structures_index_path(self):
        return self.dataset_path() / "input_structures.idx"
        

    def batches_count(self) -> int:
        """Count the number of batch directories (numeric directories only)."""
        structures_path = self.structures_path()
        if not structures_path.exists():
            return 0
        count = 0
        for item in structures_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                count += 1
        return count

    def add_client(self):
        if self._client is None:
            self._client = create_client(self.is_hpc_cluster)
            logger.debug(f"Client created: {self._client}")
        else:
            logger.info(f"Client already exists: {self._client}")

    def requested_ids(self) -> List[str]:
        if self.collection_type is CollectionType.subset:
            if self.ids_file.suffix.lower() == '.csv':
                from .csv_processor import CSVProcessor
                csv_processor = CSVProcessor(self.ids_file)
                # Detect if this is complex format by checking for double underscore or comma
                if self._is_complex_csv_format():
                    return csv_processor.extract_complex_protein_ids()
                else:
                    return csv_processor.extract_ids()
            else:
                with open(self.ids_file, "r") as f:
                    return f.read().splitlines()
        else:
            return self.get_all_ids()
    
    def _is_complex_csv_format(self) -> bool:
        """Detect if CSV file uses complex format with ranges and chains
        
        Returns:
            True if complex format detected, False otherwise
        """
        if not self.ids_file.exists():
            return False
            
        try:
            with open(self.ids_file, 'r', encoding='utf-8') as file:
                # Check first few lines for complex format indicators
                for i, line in enumerate(file):
                    if i >= 5:  # Check only first 5 lines for performance
                        break
                    line = line.strip()
                    if not line:
                        continue
                    # Look for complex format indicators: __ or , in the line
                    if '__' in line or ',' in line:
                        return True
            return False
        except Exception:
            # If we can't read the file, assume simple format
            return False
    
    def get_protein_entries_with_ranges(self):
        """Get protein entries with range information for complex processing
        
        Returns:
            List of ProteinEntry objects if CSV with ranges, None otherwise
        """
        if self.collection_type is CollectionType.subset and self.ids_file.suffix.lower() == '.csv':
            if self._is_complex_csv_format():
                from .csv_processor import CSVProcessor
                csv_processor = CSVProcessor(self.ids_file)
                return csv_processor.get_protein_entries()
        return None

    def create_dataset(self):

        log_title("Creating dataset")

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
                present_file_paths = {}
                missing_ids = self.requested_ids()
            else:
                self._handle_indexes.read_indexes("dataset")
                requested_ids = self.requested_ids()

                present_file_paths, missing_ids = (
                    self._handle_indexes.find_present_and_missing_ids(
                        "dataset", requested_ids
                    )
                )

            create_index(self.dataset_index_file_path(), present_file_paths, self.config.data_path)

            if (
                self.db_type == DatabaseType.other
                and self.collection_type == CollectionType.subset
                and len(missing_ids) > 0
            ):
                raise RuntimeError(
                    "Missing ids are not allowed when subsetting all DBs!"
                )

            if len(missing_ids) > 0:
                self.download_ids(missing_ids)
        else:
            self.download_ids(None)
        
        index = read_index(self.dataset_index_file_path(), self.config.data_path)

        if len(index.keys()) == 0:
            raise FatalDatasetError("No files found in dataset")

        self.save_dataset_metadata()

    def extract_sequence_and_coordinates(
        self, ca_mask: bool = False, substitute_non_standard_aminoacids: bool = True
    ):
        self._sequence_retriever.retrieve(self.config.disto_type)

    def get_all_ids(self):
        match self.db_type:
            case DatabaseType.PDB:
                start_time = time.time()
                all_pdbs = PDBList().get_all_entries()
                all_pdbs = map(lambda x: x.lower(), all_pdbs)
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.debug(f"PDBList().get_all_entries time: {format_time(elapsed_time)}")
                url = "http://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt"

                response = requests.get(url)
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
                res = filter_pdb_codes(response.text, all_pdbs)
                logger.debug(f"After removing non protein codes {len(res)}")
            case DatabaseType.AFDB:
                res = []
            case DatabaseType.ESMatlas:
                res = []
            case _:
                res = open(self.ids_file).readlines()
        return res

    def download_ids(self, ids):
        if self.input_path is not None:
            extracted_path = extract_archive(self.input_path, self)
            if extracted_path is not None:
                missing_ids = save_extracted_files(self, extracted_path, ids)
                # Update ids to only contain the ones that need to be downloaded
                ids = missing_ids if missing_ids is not None else []
            else:
                # If extraction fails, we should not proceed
                return

        match self.db_type:
            case DatabaseType.PDB:
                self.handle_pdb(ids)
            case DatabaseType.AFDB:
                self.handle_afdb(ids)
            case DatabaseType.ESMatlas:
                self.handle_esma()
            case DatabaseType.other:
                self

    def handle_pdb(self, ids: List[str]):
        match self.collection_type:
            case CollectionType.all:
                self._download_pdb_(ids)
            case CollectionType.part:
                pass
            case CollectionType.clust:
                pass
            case CollectionType.subset:
                self._download_pdb_(ids)


    def add_new_files_to_index(self, new_files_index):
        add_new_files_to_index(self.dataset_index_file_path(), new_files_index, self.config.data_path)

    def _download_pdb_(self, ids: List[str]):
        Path(self.structures_path()).mkdir(exist_ok=True, parents=True)
        pdb_repo_path = self.structures_path()
        
        # Get the number of existing batch directories to use as an offset
        batch_offset = self.batches_count()
        
        chunks = list(self.chunk(ids))

        mkdir_for_batches(pdb_repo_path, len(chunks), offset=batch_offset)

        logger.info(f"Downloading {len(ids)} PDBs into {len(chunks)} new chunks (offset by {batch_offset})")

        new_files_index = {}

        def run(input_data, machine):
            return self._client.submit(
                retrieve_pdb_chunk_to_h5,
                *input_data,
                self.binary_data_download,
                [machine],
                workers=[machine],
            )

        def collect(result):
            downloaded_pdbs, file_path = result
            logger.debug(f"Updating new_files_index with {len(downloaded_pdbs)} files")
            new_files_index.update({k: file_path for k in downloaded_pdbs})

        compute_batches = ComputeBatches(
            self._client,
            run,
            collect,
            "pdb" + "_b" if self.binary_data_download else "",
            len(chunks)
        )

        inputs = (
            (pdb_repo_path / f"{i + batch_offset}", ids_chunk) for i, ids_chunk in enumerate(chunks)
        )

        factor = 10
        factor = 15 if total_workers() > 1500 else factor
        factor = 20 if total_workers() > 2000 else factor
        compute_batches.compute(inputs, factor=factor)

        logger.info("Adding new files to index")

        try:
            logger.info(f"Extracted {len(new_files_index)} new protein chain(s)")
            self.add_new_files_to_index(new_files_index)
        except Exception as e:
            logger.error(f"Failed to update index: {e}")

    def _download_afdb_(self, ids: List[str]):
        # Convert AFDB IDs from format "AF-A0A009IHW8-F1-model_v4" to "A0A009IHW8"
        # Handle IDs that may come in either format
        processed_ids = []
        for file_id in ids:
            # Remove "AF-" prefix if present and "-F1-model_v4" suffix if present
            processed_id = file_id.removeprefix("AF-").removesuffix("-F1-model_v4")
            processed_ids.append(processed_id)
        
        ids = processed_ids
        logger.info(f"Processed {len(ids)} AFDB IDs for download")
        
        Path(self.structures_path()).mkdir(exist_ok=True, parents=True)
        afdb_repo_path = self.structures_path()
        
        # Get the number of existing batch directories to use as an offset
        batch_offset = self.batches_count()
        
        chunks = list(self.chunk(ids))

        mkdir_for_batches(afdb_repo_path, len(chunks), offset=batch_offset)

        logger.info(f"Downloading {len(ids)} AFDB structures into {len(chunks)} new chunks (offset by {batch_offset})")

        new_files_index = {}

        def run(input_data, machine):
            return self._client.submit(
                retrieve_afdb_chunk_to_h5,
                *input_data,
                [machine],
                workers=[machine],
            )

        def collect(result):
            downloaded_ids, file_path = result
            logger.debug(f"Updating new_files_index with {len(downloaded_ids)} files")
            new_files_index.update({k: file_path for k in downloaded_ids})

        compute_batches = ComputeBatches(
            self._client,
            run,
            collect,
            "afdb",
            len(chunks)
        )

        inputs = (
            (afdb_repo_path / f"{i + batch_offset}", ids_chunk) for i, ids_chunk in enumerate(chunks)
        )

        factor = 10
        factor = 15 if total_workers() > 1500 else factor
        factor = 20 if total_workers() > 2000 else factor
        compute_batches.compute(inputs, factor=factor)

        logger.info("Adding new files to index")

        try:
            logger.info(f"Extracted {len(new_files_index)} new protein chain(s)")
            self.add_new_files_to_index(new_files_index)
        except Exception as e:
            logger.error(f"Failed to update index: {e}")

    def _download_afdb_(self, ids: List[str]):
        # Convert AFDB IDs from format "AF-A0A009IHW8-F1-model_v4" to "A0A009IHW8"
        # Handle IDs that may come in either format
        processed_ids = []
        for file_id in ids:
            # Remove "AF-" prefix if present and "-F1-model_v4" suffix if present
            processed_id = file_id.removeprefix("AF-").removesuffix("-F1-model_v4")
            processed_ids.append(processed_id)
        
        ids = processed_ids
        logger.info(f"Processed {len(ids)} AFDB IDs for download")
        
        Path(self.structures_path()).mkdir(exist_ok=True, parents=True)
        afdb_repo_path = self.structures_path()
        chunks = list(self.chunk(ids))

        mkdir_for_batches(afdb_repo_path, len(chunks))

        logger.info(f"Downloading {len(ids)} AFDB structures into {len(chunks)} chunks")

        new_files_index = {}

        def run(input_data, machine):
            return self._client.submit(
                retrieve_afdb_chunk_to_h5,
                *input_data,
                [machine],
                workers=[machine],
            )

        def collect(result):
            downloaded_ids, file_path = result
            logger.debug(f"Updating new_files_index with {len(downloaded_ids)} files")
            new_files_index.update({k: file_path for k in downloaded_ids})

        compute_batches = ComputeBatches(
            self._client,
            run,
            collect,
            "afdb",
            len(chunks)
        )

        inputs = (
            (afdb_repo_path / f"{i}", ids_chunk) for i, ids_chunk in enumerate(chunks)
        )

        factor = 10
        factor = 15 if total_workers() > 1500 else factor
        factor = 20 if total_workers() > 2000 else factor
        compute_batches.compute(inputs, factor=factor)

        logger.info("Adding new files to index")

        try:
            logger.info(f"Extracted {len(new_files_index)} new protein chain(s)")
            self.add_new_files_to_index(new_files_index)
        except Exception as e:
            logger.error(f"Failed to update index: {e}")

    def foldcomp_decompress(self):

        db_path = str(self.dataset_repo_path() / self.proteome)

        ids = (
            db.read_text(f"{self.dataset_repo_path()}/{self.proteome}.lookup")
            .filter(
                lambda line: (
                    ".pdb" in line if (".pdb" in line or ".cif" in line) else True
                )
            )
            .map(lambda line: line.split()[1])
            .compute()
        )

        if self.ids_file:
            subset_ids = self.ids_file.read_text().splitlines()
            # Check if subset_id is included in any of the full ids
            ids = [id for id in ids if any(subset_id in id for subset_id in subset_ids)]

        structures_path = self.structures_path()
        structures_path.mkdir(exist_ok=True, parents=True)

        batches = list(self.chunk(ids))
        mkdir_for_batches(structures_path, len(batches))

        futures = [
            self._client.submit(
                alphafold_chunk_to_h5,
                db_path,
                str(structures_path / f"{number}"),
                list(batch),
            )
            for number, batch in enumerate(batches)
        ]

        result_index = {}

        i = 0
        total = len(futures)

        for batch in as_completed(futures, with_results=True).batches():
            for _, single_batch_index in batch:
                result_index.update(single_batch_index)
                logger.debug(f"Processing batch {i}/{total}")
                i += 1

        create_index(self.dataset_index_file_path(), result_index, self.config.data_path)

    def handle_afdb(self, ids: List[str]):
        match self.collection_type:
            case CollectionType.all:
                pass
            case CollectionType.part:
                foldcomp_download(self.proteome, str(self.dataset_repo_path()))
                self.foldcomp_decompress()
            case CollectionType.clust:
                foldcomp_download(self.proteome, str(self.dataset_repo_path()))
                self.foldcomp_decompress()
            case CollectionType.subset:
                if ids:
                    self._download_afdb_(ids)
                else:
                    logger.info("No missing IDs to download for AFDB subset")

    def handle_esma(self):
        match self.collection_type:
            case CollectionType.all:
                pass
            case CollectionType.part:
                pass
            case CollectionType.clust:
                foldcomp_download(self.proteome, str(self.dataset_repo_path()))
                self.foldcomp_decompress()
            case CollectionType.subset:
                pass

    

    def handle_archive(self):
        """Handle creating dataset from tar.gz archive"""
        if not self.archive_path or not self.archive_path.exists():
            raise ValueError("Archive path must be provided and exist")
        
        # Create temporary extraction directory
        temp_dir = self.dataset_repo_path() / "temp"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Extract archive
            with tarfile.open(self.archive_path, 'r:gz') as tar:
                tar.extractall(path=temp_dir)
            
            # Get all files from temp directory
            files = list(temp_dir.glob('*.cif'))
            total_files = len(files)
            
            if total_files == 0:
                raise ValueError(f"No .cif files found in archive {self.archive_path}")
                
            # Calculate number of batches needed
            num_batches = ceil(total_files / self.batch_size)
            
            # Create structures directory
            structures_path = self.structures_path()
            structures_path.mkdir(exist_ok=True, parents=True)
            
            # Create batch directories
            for i in range(num_batches):
                (structures_path / str(i)).mkdir(exist_ok=True)

            # Process batches in parallel using dask
            new_files_index = {}
            
            def run(input_data, machine):
                batch_files, batch_dir = input_data
                return self._client.submit(extract_batch_from_archive, batch_files, batch_dir, workers=[machine])

            def collect(result):
                pdbs, h5_path = result
                if pdbs and h5_path:
                    new_files_index.update({pdb: h5_path for pdb in pdbs})

            compute_batches = ComputeBatches(self._client, run, collect, "archive_processing")

            # Create input data for each batch
            inputs = (
                (files[i:i + self.batch_size], structures_path / str(i // self.batch_size))
                for i in range(0, total_files, self.batch_size)
            )

            # Process batches
            compute_batches.compute(inputs)
            
            logger.info(f"Created {num_batches} batches with up to {self.batch_size} files each")
            
            # Update index
            if new_files_index:
                logger.info(f"Adding new files to index, len: {len(new_files_index)}")
                create_index(self.dataset_index_file_path(), new_files_index, self.config.data_path)
            
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def chunk(self, it):
        return list(chunk(it, self.batch_size))

    def save_dataset_metadata(self):
        with (self.dataset_path() / "dataset.json").open("w+") as json_dataset_file:
            json_dataset_file.write(self.model_dump_json())

    def generate_distograms(self):
        """Generate distograms for the dataset."""
        generate_distograms(self)

    def generate_embeddings(self):
        """Generate embeddings for the dataset."""
        embedding = Embedding(self)
        embedding.run()


def create_subset():
    StructuresDataset(
        db_type=DatabaseType.other,
        collection_type=CollectionType.subset,
        ids_file=Path("./mix_ids1.txt"),
    ).create_dataset()


def create_e_coli():
    StructuresDataset(
        db_type=DatabaseType.AFDB,
        collection_type=CollectionType.part,
        proteome="e_coli",
    ).create_dataset()


def create_swissprot():
    StructuresDataset(
        db_type=DatabaseType.AFDB,
        collection_type=CollectionType.part,
        proteome="afdb_swissprot_v4",
    ).create_dataset()


if __name__ == "__main__":
    # create_swissprot()
    # create_e_coli()

    dataset = StructuresDataset(
        db_type=DatabaseType.PDB,
        collection_type=CollectionType.all,
        # ids_file=Path("/Users/youngdashu/sano/deepFRI2-toolbox-dev/ids_test.txt")
    )

    dataset.create_dataset()

    # create_e_coli()

    pass
