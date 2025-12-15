from __future__ import annotations

import logging
import os


def setup_logging(level: str | int | None = None) -> None:
    if level is None:
        level = os.getenv("MOPE_LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


logger = logging.getLogger("mope")
