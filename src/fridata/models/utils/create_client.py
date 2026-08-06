import os

import dask
from dask.distributed import LocalCluster, Client

import warnings
import distributed

import logging

from fridata.utlis.logging import logger

dask.config.set({"distributed.scheduler.locks.lease-timeout": "10m"})


def total_workers():
    total_cores = os.environ.get("SLURM_CPUS_PER_TASK")
    if total_cores is None:
        total_cores = os.cpu_count()
    total_nodes = os.environ.get("SLURM_NTASKS")
    if total_nodes is None:
        total_nodes = 1

    return (int(total_cores) * int(total_nodes)) - 2


def get_cluster_machines(client):
    scheduler_info = client.scheduler_info()
    workers = scheduler_info["workers"]
    machines = set(worker_info["host"] for worker_info in workers.values())
    return machines


class WorkerLogFilter(logging.Filter):
    """Filter to suppress verbose worker startup INFO logs."""
    
    def filter(self, record):
        message = record.getMessage()
        # Filter out verbose worker startup messages
        if "Starting established connection to tcp://" in message:
            return False
        if "Start worker at:" in message:
            return False
        if "Start Nanny at:" in message:
            return False
        if "Listening to:" in message:
            return False
        if "Worker name:" in message:
            return False
        if "dashboard at:" in message:
            return False
        if "Waiting to connect to:" in message:
            return False
        if "Threads:" in message:
            return False
        if "Memory:" in message:
            return False
        if "Local Directory:" in message:
            return False
        if "Register worker" in message:
            return False
        # Filter out separator lines
        if message.strip() == "-" * len(message.strip()) and len(message.strip()) > 10:
            return False
        return True


def create_client(is_slurm_client: bool):
    # Configure logging before creating the client
    # Use a more robust logging configuration that won't conflict with pytest
    distributed_logger = logging.getLogger('distributed')
    distributed_logger.setLevel(logging.WARNING)
    
    # Add filter to suppress verbose worker startup logs
    worker_log_filter = WorkerLogFilter()
    
    # Apply filter to distributed logger and its handlers
    distributed_logger.addFilter(worker_log_filter)
    
    # Also apply to worker-specific logger
    worker_logger = logging.getLogger('distributed.worker')
    worker_logger.addFilter(worker_log_filter)

    
    # Add a null handler to prevent "No handlers could be found" warnings
    # and to avoid conflicts with test cleanup
    if not distributed_logger.handlers:
        null_handler = logging.NullHandler()
        distributed_logger.addHandler(null_handler)

    logging.getLogger('distributed.worker').setLevel(logging.WARNING)

    
    # Silence specific Dask warnings
    warnings.simplefilter("ignore", distributed.comm.core.CommClosedError)
    ignore_list = [
        ".*Creating scratch directories is taking a surprisingly long time.*",
        "Failed to communicate with scheduler during heartbeat.*",
        "Connection to tcp:*has been closed.*",
        
    ]   
    for warning in ignore_list:
        warnings.filterwarnings(
            "ignore",
            message=warning,
        )
    

    if is_slurm_client:
        client = Client(
            scheduler_file=os.environ.get("DEEPFRI_PATH") + "/scheduler.json"
        )
        n = total_workers()
        logger.info("Creating Dask computation client with {} workers".format(n))
        client.wait_for_workers(n, 300.0)
    else:
        total_cores = os.cpu_count()
        logger.info("Creating Dask computation client with {} workers".format(total_cores - 1))
        # Create a LocalCluster with the calculated number of workers
        cluster = LocalCluster(
            dashboard_address="0.0.0.0:8991",  # Use different port to avoid conflicts
            n_workers=total_cores - 1,
            threads_per_worker=1,
            memory_limit="100 GiB",
            silence_logs=logging.CRITICAL,  # Silence all logs including cleanup
        )

        client = Client(cluster)
    logger.debug(f"Dashboard link: {client.dashboard_link}")
    logger.debug(f"Workers count: {len(client.scheduler_info()['workers'])}")
    logger.debug(f"Machines: {get_cluster_machines(client)}")

    return client
