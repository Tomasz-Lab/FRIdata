
from functools import reduce
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

from dask import delayed
from distributed import Client, progress

from fridata.models.manage_dataset.database_type import DatabaseType
from fridata.models.manage_dataset.index.handle_index import read_index
from fridata.scripts.list_datasets import SEPARATOR


def search_index(path: Path, ids_searched_for: List[str], data_path: str):
    index = read_index(path, data_path)

    res = {}

    for id_ in ids_searched_for:
        if id_ in index:
            res[id_] = index[id_]

    return res


def search_indexes(
    structures_dataset,
    base_path: Path,
    batched_ids: Iterable[List[str]],
    index_file_name: str,
) -> Tuple[Dict[str, str], List[str]]:
    db_type = structures_dataset.db_type
    if db_type == DatabaseType.other:
        glob_pattern = f"**/*/{index_file_name}.idx"
    else:
        glob_pattern = f"**/{db_type.name}{SEPARATOR}*/{index_file_name}.idx"
    sequence_indexes = base_path.glob(glob_pattern)

    tasks = [
        delayed(search_index)(file, batch, structures_dataset.config.data_path)
        for batch in batched_ids
        for file in sequence_indexes
    ]

    futures = structures_dataset._client.compute(tasks)
    progress(futures)
    results = structures_dataset._client.gather(futures)

    results = reduce(lambda a, b: {**a, **b}, results, {})
    return results, find_missing_ids(results, batched_ids)


def find_missing_ids(index: Dict[str, str], ids: Iterable[List[str]]) -> List[str]:
    return [id_ for batch in ids for id_ in batch if id_ not in index]
