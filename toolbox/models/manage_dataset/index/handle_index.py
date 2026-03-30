import json
from pathlib import Path
from typing import Dict, Union, List

from toolbox.models.manage_dataset.utils import groupby_dict_by_values
from toolbox.utlis.logging import logger


def read_index(index_file_path: Path, data_path: str) -> Dict[str, str]:
    try:
        with index_file_path.open('r') as f:
            index = json.load(f)
            if "reversed" in index_file_path.stem:
                index = {data_path + "/" + k: v for k, v in index.items()}
            else:
                index = {k: data_path + "/" + v for k, v in index.items()}
            return index
    except FileNotFoundError:
        resolved = index_file_path.resolve()
        if index_file_path.name == "dataset.idx":
            logger.error(
                "Dataset index not found: %s. Run generate_data with 'dataset' in -t "
                "(e.g. -t dataset,sequences,...) or run create_dataset first so the structure index is created.",
                resolved,
            )
        else:
            logger.error("Index file not found: %s", resolved)
        return {}
    except Exception:
        logger.exception("read_index failed for %s", index_file_path)
        return {}


def create_index(index_file_path: Path, values: Union[Dict[str, str], List[str]], data_path: str):
    with index_file_path.open("w") as f:
        if isinstance(values, list):
            values = {str(i): v for i, v in enumerate(values)}
        # Map over values to ensure they are strings
        values = {k: v.removeprefix(data_path + "/") for k, v in values.items()}
        json.dump(values, f)

    file_name = index_file_path.stem

    values = groupby_dict_by_values(values)
    with (index_file_path.parent / f"{file_name}_reversed.idx").open("w") as f:
        json.dump(values, f)


def add_new_files_to_index(dataset_index_file_path: Path, new_files_index: Dict, data_path: str):
    try:
        current_index = read_index(dataset_index_file_path, data_path)
        current_index.update(new_files_index)
        create_index(dataset_index_file_path, current_index, data_path)
    except Exception:
        logger.exception("add_new_files_to_index failed for %s with error: %s", dataset_index_file_path, e)
