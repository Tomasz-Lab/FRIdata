"""
Regenerate expected test data files.

This script runs all test scenarios to generate new expected outputs
that will be used for test comparisons. Use this when embedding generation
or other core functionality changes.

Usage:
    python tests/regenerate_expected_data.py

Outputs:
    - Backs up current expected data to: tests/data/dataset_expected_backup_<timestamp>
    - Generates new expected data to: tests/data/dataset_expected_backup_2
"""

import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from dask.distributed import Client, LocalCluster
import logging
import warnings
import distributed

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from toolbox.config import Config
from toolbox.models.embedding.embedder.embedder_type import EmbedderType
from toolbox.models.manage_dataset.structures_dataset import StructuresDataset
from toolbox.models.manage_dataset.collection_type import CollectionType
from toolbox.models.manage_dataset.database_type import DatabaseType

# Set permissions for new files
os.umask(0o002)

# Paths
TEST_DATA_DIR = Path(__file__).parent / "data"
EXPECTED_DIR = TEST_DATA_DIR / "dataset_expected"
OUTPUT_DIR_NAME = "dataset_expected_backup_2"
OUTPUT_DIR = TEST_DATA_DIR / OUTPUT_DIR_NAME
INPUT_LISTS_DIR = EXPECTED_DIR / "input_lists"
# Final path that will be used in the config (what tests expect for generated data)
FINAL_EXPECTED_PATH = TEST_DATA_DIR / "dataset_generated"

# Configure distributed logging
distributed_logger = logging.getLogger('distributed')
distributed_logger.setLevel(logging.WARNING)
if not distributed_logger.handlers:
    null_handler = logging.NullHandler()
    distributed_logger.addHandler(null_handler)

# Silence Dask warnings
warnings.simplefilter("ignore", distributed.comm.core.CommClosedError)
warnings.filterwarnings(
    "ignore",
    message=".*Creating scratch directories is taking a surprisingly long time.*",
)

def backup_current_expected_data():
    """Backup current expected data with timestamp."""
    if not EXPECTED_DIR.exists():
        print(f"⚠️  No existing expected data found at {EXPECTED_DIR}")
        return None

    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    backup_dir = TEST_DATA_DIR / f"dataset_expected_backup_{timestamp}"

    print(f"📦 Backing up current expected data...")
    print(f"   From: {EXPECTED_DIR}")
    print(f"   To:   {backup_dir}")

    shutil.copytree(EXPECTED_DIR, backup_dir)
    print(f"✅ Backup completed: {backup_dir}")
    return backup_dir

def setup_dask_cluster():
    """Setup Dask cluster for parallel processing."""
    n_cores = os.cpu_count()
    print(f"🔧 Setting up Dask cluster with {n_cores - 1} workers...")

    cluster = LocalCluster(
        n_workers=n_cores - 1,
        threads_per_worker=1,
        memory_limit="4 GiB",
        silence_logs=logging.CRITICAL,
        worker_dashboard_address=None,
        dashboard_address="0.0.0.0:8990",
    )
    client = Client(cluster)
    print(f"✅ Dask cluster ready: {client}")
    return client, cluster

def cleanup_dask_cluster(client, cluster):
    """Clean up Dask client and cluster."""
    try:
        if client and client.status != "closed":
            client.close()
        if cluster:
            cluster.close()
        print("✅ Dask cluster cleaned up")
    except Exception as e:
        print(f"⚠️  Error cleaning up Dask cluster: {e}")

def create_dataset_and_abstractions(dataset_name, ids_file_path, overwrite, client):
    """Create dataset with all abstractions (sequences, distograms, embeddings)."""

    config = Config(
        data_path=str(OUTPUT_DIR),
        disto_type="CA",
        disto_thr="inf",
        separator="-",
        batch_size=1000,
    )

    print(f"\n{'='*70}")
    print(f"📊 Creating dataset: {dataset_name}")
    print(f"{'='*70}")
    print(f"Input file: {ids_file_path}")
    print(f"Overwrite: {overwrite}")
    print(f"Config: {config.model_dump_json(indent=2)}")

    dataset = StructuresDataset(
        db_type=DatabaseType.PDB,
        collection_type=CollectionType.subset,
        version=dataset_name,
        ids_file=ids_file_path,
        overwrite=overwrite,
        config=config,
        embedder_type=EmbedderType.ESM2_T33_650M,
    )
    dataset._client = client

    # Execute pipeline with timing
    operations = [
        ("create_dataset", dataset.create_dataset),
        ("extract_sequence_and_coordinates", dataset.extract_sequence_and_coordinates),
        ("generate_distograms", dataset.generate_distograms),
        ("generate_embeddings", dataset.generate_embeddings),
    ]

    total_time = 0
    for op_name, op_func in operations:
        print(f"\n▶️  Running: {op_name}...")
        start_time = time.time()
        op_func()
        elapsed = time.time() - start_time
        total_time += elapsed
        print(f"✅ {op_name} completed in {elapsed:.2f}s")

    print(f"\n{'='*70}")
    print(f"✅ Dataset '{dataset_name}' completed in {total_time:.2f}s")
    print(f"{'='*70}\n")

def main():
    """Main regeneration script."""
    print("\n" + "="*70)
    print("🔄 EXPECTED TEST DATA REGENERATION SCRIPT")
    print("="*70)

    # Step 1: Backup current expected data
    print("\n[STEP 1] Backup current expected data")
    backup_dir = backup_current_expected_data()

    # Step 2: Prepare output directory
    print("\n[STEP 2] Prepare output directory")
    if OUTPUT_DIR.exists():
        print(f"⚠️  Output directory exists, removing: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    print(f"✅ Output directory ready: {OUTPUT_DIR}")

    # Step 3: Setup Dask cluster
    print("\n[STEP 3] Setup Dask cluster")
    client, cluster = setup_dask_cluster()

    try:
        # Step 4: Run all test scenarios
        print("\n[STEP 4] Run all test scenarios")

        test_scenarios = [
            ("initial_5", "pdb_5.txt", False, "Create initial dataset with 5 PDB structures"),
            ("second_5", "pdb_5.txt", True, "Create second dataset with 5 PDB structures (overwrite)"),
            ("third_7", "pdb_7.txt", False, "Create third dataset with 7 PDB structures (2 new + 5 reused)"),
            ("fourth_7", "pdb_7.txt", True, "Create fourth dataset with 7 PDB structures (overwrite)"),
            ("fifth_7", "pdb_7.txt", False, "Create fifth dataset with 7 PDB structures (all reused)"),
        ]

        overall_start = time.time()

        for i, (dataset_name, ids_file, overwrite, description) in enumerate(test_scenarios, 1):
            print(f"\n{'#'*70}")
            print(f"# TEST {i}/5: {dataset_name}")
            print(f"# {description}")
            print(f"{'#'*70}")

            ids_file_path = INPUT_LISTS_DIR / ids_file
            create_dataset_and_abstractions(dataset_name, ids_file_path, overwrite, client)

        overall_elapsed = time.time() - overall_start

        # Copy input_lists directory to output
        print("\n[STEP 5] Copy input_lists directory")
        output_input_lists = OUTPUT_DIR / "input_lists"
        if INPUT_LISTS_DIR.exists():
            shutil.copytree(INPUT_LISTS_DIR, output_input_lists)
            print(f"✅ Copied input_lists from {INPUT_LISTS_DIR} to {output_input_lists}")
        else:
            print(f"⚠️  Warning: input_lists directory not found at {INPUT_LISTS_DIR}")

        # Fix data_path in all dataset.json files
        print("\n[STEP 6] Fix data_path in dataset.json files")
        import json
        dataset_json_files = list((OUTPUT_DIR / "datasets").rglob("dataset.json"))
        print(f"Found {len(dataset_json_files)} dataset.json files to update")

        for json_file in dataset_json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Update data_path from OUTPUT_DIR to FINAL_EXPECTED_PATH
            if 'config' in data and 'data_path' in data['config']:
                old_path = data['config']['data_path']
                data['config']['data_path'] = str(FINAL_EXPECTED_PATH)

                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"  Updated {json_file.relative_to(OUTPUT_DIR)}")
                print(f"    {old_path} -> {data['config']['data_path']}")

        print(f"✅ Fixed data_path in {len(dataset_json_files)} files")

        # Summary
        print("\n" + "="*70)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Total execution time: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f}m)")
        print(f"\nGenerated outputs location:")
        print(f"  {OUTPUT_DIR}")
        if backup_dir:
            print(f"\nBackup location:")
            print(f"  {backup_dir}")
        print("\n📋 Next steps:")
        print("  1. Manually verify the generated outputs")
        print("  2. Compare with current expected files if needed")
        print("  3. Replace tests/data/dataset_expected with dataset_expected_backup_2")
        print("     (e.g., rm -rf tests/data/dataset_expected && mv tests/data/dataset_expected_backup_2 tests/data/dataset_expected)")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR during regeneration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Step 7: Cleanup
        print("\n[STEP 7] Cleanup Dask cluster")
        cleanup_dask_cluster(client, cluster)

    return 0

if __name__ == "__main__":
    exit(main())
