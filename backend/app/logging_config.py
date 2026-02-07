"""Centralized logging configuration for ContractIQ — all logs to ./tmp/."""

import logging
import os
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure root logger with console + optional file handler.

    Args:
        log_level: Standard Python logging level name.
        log_file: If provided, logs are also written to ./tmp/<log_file>.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(fmt)
        root.addHandler(console)

    if log_file:
        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        log_path = tmp_dir / log_file

        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path.resolve()) for h in root.handlers):
            fh = logging.FileHandler(log_path)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            root.addHandler(fh)
