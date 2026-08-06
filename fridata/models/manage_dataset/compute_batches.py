import threading
import time

from itertools import cycle
from typing import Any, Callable, Generator, Optional, Tuple

from fridata.models.manage_dataset.utils import format_time

from dask.distributed import Client, Semaphore, as_completed, Future
from distributed import Variable

from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

from fridata.models.utils.create_client import total_workers, get_cluster_machines

from fridata.utlis.logging import logger


class ComputeBatches:

    def __init__(self, client: Client, run_f, collect_f, name: str, inputs_len: Optional[int] = None):
        self.client = client
        self.run_f: Callable[[Any, str], Future] = run_f
        self.collect_f: Callable[[Any], None] = collect_f
        self.name: str = name
        self.inputs_len: Optional[int] = inputs_len

    def compute(self, inputs: Generator[Tuple[Any], Any, None], factor=1):

        start = time.time()

        machines = get_cluster_machines(self.client)
        machines_c = cycle(machines)

        max_workers = max(self._workers_num_() // factor, 1)
        sem_name = "sem" + self.name
        semaphore = Semaphore(max_leases=max_workers, name=sem_name)

        logger.debug(f"Max parallel workers {max_workers}")

        var = Variable("stopping-criterion")
        var.set(False)

        i = 1

        ac = as_completed([], with_results=True)

        collect_thread = threading.Thread(
            target=collect, args=(ac, self.collect_f, semaphore, self.name, self.inputs_len)
        )
        collect_thread.start()

        while True:
            semaphore.acquire()

            next_value = next(inputs, None)
            if next_value is None:
                break

            logger.debug(f"Processing batch {i}")
            i += 1
            future = self.run_f(next_value, next(machines_c))
            ac.add(future)

        var.set(True)

        collect_thread.join()

        while semaphore.get_value() > 0:
            semaphore.release()
            time.sleep(0.1)
        
        semaphore.close()

        end = time.time()

        logger.debug(f"Time taken {self.name}: {format_time(end - start)}")
        
    def _workers_num_(self):
        return total_workers()


def collect(ac: as_completed, collect_f, semaphore: Semaphore, computation_name: str, inputs_len: Optional[int] = None):
    logger.info("Collecting results")
    logger.debug("Collecting Dask results collection started")
    total_time = 0
    stop_var = Variable("stopping-criterion")
    with logging_redirect_tqdm():
        pbar = tqdm(total=inputs_len)
        while True:

            while ac.is_empty() and not stop_var.get():
                if ac.is_empty() and stop_var.get():
                    break
                time.sleep(1)

            if ac.is_empty() and stop_var.get():
                break

            future_c, result = next(ac)
            start_time = time.time()
            try:
                collect_f(result)
            except Exception as e:
                logger.error(f"Error in collect_f: {e}")
            finally:
                end_time = time.time()
                total_time += end_time - start_time
                # Ensure release happens even if collect_f fails
                del future_c
                pbar.update(1)
                semaphore.release()

            if ac.is_empty() and stop_var.get():
                break

        pbar.close()

    logger.debug(f"Collect results time: {format_time(total_time)}")
    logger.debug("Dask results collection finished")
