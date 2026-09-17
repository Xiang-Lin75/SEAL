"""Repository-root launcher for the SEAL trainer."""

from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    run_path(
        str(Path(__file__).resolve().parent / "entrypoints" / "train.py"),
        run_name="__main__",
    )
