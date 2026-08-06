import re
from pathlib import Path
from fridata.config import load_config

config = load_config()
separator = config.separator
data_path = config.data_path

if __name__ == "__main__":

    dataset_path = f"{data_path}/datasets"
    pattern = f".*\\{separator}.*\\{separator}.*\\{separator}.*"
    pattern = re.compile(pattern)
    dataset_paths = Path(dataset_path).glob("*")
    for path in dataset_paths:
        if path.is_dir() and pattern.match(path.name) is not None:
            print(path.name)
            dataset_name = path.name
            dataset_name = dataset_name.split(separator)
            print(
                f"DB:{dataset_name[0]} type:{dataset_name[1]} type_str:{dataset_name[2]} version:{dataset_name[3]}"
            )
            print("\t\t\t\t")
