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

sys.path.append(str(data_path))

here = pathlib.Path(__file__).resolve()
repo_root = here.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))