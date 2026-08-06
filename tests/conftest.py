import pytest
import dotenv
import logging
import sys
from _pytest.monkeypatch import MonkeyPatch
from fridata.utlis.logging import set_config
import os

from tests.paths import OUTPATH, EXPPATH


@pytest.fixture(scope="module", autouse=True)
def setup_main_path(request):

    sys._called_from_test = True

    dotenv.load_dotenv()
    mp = MonkeyPatch()
    if os.path.basename(request.module.__file__) == "test_handle_indexes.py":
        mp.setenv("DATA_PATH", str(EXPPATH))
    else:
        mp.setenv("DATA_PATH", str(OUTPATH))
    
    def configure_logging(logging_module):
        # Enhanced logging configuration for performance monitoring
        logging_module.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging_module.StreamHandler(sys.stdout),
                logging_module.FileHandler('test_performance.log', mode='w')
            ],
            force=True  # Override any existing logging config
        )
        
        # Set specific loggers to INFO level to capture performance data
        logging_module.getLogger('fridata').setLevel(logging.INFO)
        logging_module.getLogger('fridata.models').setLevel(logging.INFO)
        logging_module.getLogger('fridata.models.manage_dataset').setLevel(logging.INFO)
        
    set_config(configure_logging)

    yield
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def cleanup_dask():
    """Ensure all Dask clients are properly closed after tests"""
    yield
    
    # Clean up any remaining Dask clients and clusters
    try:
        from dask.distributed import Client, LocalCluster
        import gc
        
        # Suppress noisy tornado/asyncio shutdown tracebacks
        logging.getLogger('tornado.application').setLevel(logging.CRITICAL)
        logging.getLogger('distributed.nanny').setLevel(logging.CRITICAL)
        logging.getLogger('distributed.process').setLevel(logging.CRITICAL)
        
        # Close any active clients
        for client in list(Client._instances):
            try:
                if client.status != "closed":
                    # Try to close the cluster if it's a LocalCluster
                    if hasattr(client, 'cluster') and isinstance(client.cluster, LocalCluster):
                        client.cluster.close()
                    client.close()
            except Exception:
                # Ignore individual client cleanup errors
                pass
        
        # Clear the instances list
        Client._instances.clear()
        
        # Force garbage collection to clean up any lingering objects
        gc.collect()
        
    except Exception:
        # Ignore cleanup errors
        pass