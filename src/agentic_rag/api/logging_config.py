"""
logging_config.py
==================
One place to configure logging for the whole API process, instead of
scattered print()/basicConfig() calls.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. reload worker re-import) — avoid duplicate handlers.
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quiet noisy third-party loggers unless we're at DEBUG.
    if level.upper() != "DEBUG":
        for noisy in ("httpx", "httpcore", "qdrant_client"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
