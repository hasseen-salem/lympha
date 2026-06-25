from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_LOG_SIZE = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def setup_logger(
    name: str = "lympha",
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    if log_file is None:
        log_file = (
            Path(__file__).resolve().parent.parent.parent
            / "logs"
            / "detector.log"
        )
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=_MAX_LOG_SIZE,
        backupCount=_BACKUP_COUNT,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
