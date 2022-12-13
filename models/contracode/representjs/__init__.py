import os
from pathlib import Path
def get_package_root():

    return Path(os.path.dirname(os.path.abspath(__file__))) / ".."


PACKAGE_ROOT = get_package_root()
DATA_DIR = get_package_root() / "data"
CSNJS_DIR = DATA_DIR / "codesearchnet_javascript"
RUN_DIR = Path(os.path.dirname("/outputs/contracode/runs"))

DATA_DIR.mkdir(exist_ok=True)
RUN_DIR.mkdir(exist_ok=True)
