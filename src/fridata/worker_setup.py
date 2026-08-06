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

# parents[1] is the directory that holds the `fridata` package (src/ in a
# checkout, site-packages when installed). Putting it on sys.path lets dask
# workers `import fridata` when this file is used as a --preload script.
here = pathlib.Path(__file__).resolve()
package_parent = here.parents[1]
if str(package_parent) not in sys.path:
    sys.path.insert(0, str(package_parent))