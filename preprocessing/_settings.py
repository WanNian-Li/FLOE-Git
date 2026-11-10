"""Portable defaults shared by standalone preprocessing entry points."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get(
    "FLOE_PREPROCESS_ROOT",
    str(Path(__file__).resolve().parent.parent / "data" / "floe_preprocessing"),
)).expanduser().resolve()
