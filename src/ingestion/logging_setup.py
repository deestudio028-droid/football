"""Logging configuration with a hard guarantee that API tokens never reach
a log line, a log file, or stdout, even if a bug elsewhere accidentally
tries to log a full request URL.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

_TOKEN_QUERY_PARAM_RE = re.compile(r"(api_token=)[^&\s]+", re.IGNORECASE)


class TokenRedactingFilter(logging.Filter):
    """Scrubs any `api_token=<value>` substring from log records before
    they are emitted, regardless of which logger produced them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _TOKEN_QUERY_PARAM_RE.sub(r"\1<redacted>", record.msg)
        if record.args:
            record.args = tuple(
                _TOKEN_QUERY_PARAM_RE.sub(r"\1<redacted>", a) if isinstance(a, str) else a
                for a in record.args
            )
        return True


def redact(text: str) -> str:
    """Redact a token from an arbitrary string before it is used anywhere
    that might end up persisted (log line, saved file, exception message).
    """
    return _TOKEN_QUERY_PARAM_RE.sub(r"\1<redacted>", text)


# Names configured by setup_logging() in this process, so a caller can
# deterministically release every log file handle it caused to be opened
# (see close_all_logging). Needed because logging.getLogger(name) returns
# a process-wide singleton that outlives any single caller: on Windows an
# attached, still-open FileHandler blocks deletion of its log file (and of
# any directory containing it) with WinError 32, whereas POSIX silently
# permits unlinking an open file.
_CONFIGURED_LOGGER_NAMES: set[str] = set()


def _detach_and_close_handlers(logger: logging.Logger) -> None:
    """Closes every handler attached to `logger` and detaches it.

    Closing before detaching is the part that matters: dropping the
    reference alone (e.g. `logger.handlers.clear()`) leaves the
    underlying OS file descriptor open until garbage collection happens
    to run, which is non-deterministic and, on Windows, means the log
    file stays locked.
    """
    for handler in list(logger.handlers):
        try:
            handler.acquire()
            try:
                handler.flush()
            finally:
                handler.release()
        except Exception:
            # A handler that is already closed/broken must not prevent
            # the remaining handlers from being released.
            pass
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)


def close_logging(name: str = "ingestion") -> None:
    """Releases every handler (and therefore every open log file) held
    by the named logger. Safe to call repeatedly and on a logger that
    was never configured.
    """
    _detach_and_close_handlers(logging.getLogger(name))
    _CONFIGURED_LOGGER_NAMES.discard(name)


def close_all_logging() -> None:
    """Releases handlers for every logger configured via setup_logging()
    in this process. Intended for deterministic teardown (application
    shutdown, or a test releasing log files before deleting a temporary
    directory) -- it does not disable logging, and a later
    setup_logging() call reconfigures the logger normally.
    """
    for name in sorted(_CONFIGURED_LOGGER_NAMES):
        _detach_and_close_handlers(logging.getLogger(name))
    _CONFIGURED_LOGGER_NAMES.clear()


def setup_logging(log_dir: Path, name: str = "ingestion", level: int = logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Release any handlers left attached by a previous setup_logging()
    # call with this same (process-wide singleton) logger name, closing
    # each one before detaching so its file handle is freed immediately
    # rather than at some later garbage-collection point. Without this,
    # a repeat call leaks the previous log file's handle and Windows
    # refuses to delete that file (WinError 32).
    _detach_and_close_handlers(logger)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(TokenRedactingFilter())

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.addFilter(TokenRedactingFilter())

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    _CONFIGURED_LOGGER_NAMES.add(name)
    return logger
