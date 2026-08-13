from __future__ import annotations

import logging
from pathlib import Path


class OnceFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        if not getattr(record, "once", False):
            return True

        key = (record.name, record.levelno, record.msg, record.args)

        if key in self.seen:
            return False

        self.seen.add(key)
        return True


def setup_logging(
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    for handler in handlers:
        handler.addFilter(OnceFilter())

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )
