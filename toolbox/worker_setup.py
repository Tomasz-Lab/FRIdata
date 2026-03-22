import sys
import os
import dotenv
import pathlib
import warnings

warnings.filterwarnings(
    "ignore", message="Creating scratch directories is taking a surprisingly long time."
)

dotenv.load_dotenv()
data_path = os.getenv("DATA_PATH")
data_path = pathlib.Path(data_path).parent / "fridata"

sys.path.append(str(data_path))
