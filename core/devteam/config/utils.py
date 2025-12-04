from pathlib import Path
from os import path

def get_relative_path(relative_path: str) -> Path:
    curent_dir = Path(__file__).parent
    return Path(path.join(curent_dir, relative_path)).resolve()