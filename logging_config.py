from __future__ import annotations

import logging
import os
import sys
import threading
from collections import deque
from datetime import datetime
from typing import Iterable, List

_DEFAULT_LEVEL = os.getenv("ARCTICROUTE_LOG_LEVEL", "INFO").upper()
_RESOLVED_LEVEL = logging.getLevelName(_DEFAULT_LEVEL)
if isinstance(_RESOLVED_LEVEL, str):
    _RESOLVED_LEVEL = logging.INFO

_BUFFER_SIZE = int(os.getenv("ARCTICROUTE_LOG_BUFFER", "500"))
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(run_id)s | ArcticRoute.%(module)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_buffer_lock = threading.Lock()
_log_buffer: deque[str] = deque(maxlen=_BUFFER_SIZE)
_run_id_lock = threading.Lock()
_CURRENT_RUN_ID = "-"


class RunIdFilter(logging.Filter):
    """Inject the run_id into each log record."""
    def filter(self, record):
        record.run_id = _CURRENT_RUN_ID
        return True


class _BufferingHandler(logging.Handler):
    """Capture log records into an in-memory ring buffer."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self._formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    def emit(self, record: logging.LogRecord) -> None:
        message = self._formatter.format(record)
        with _buffer_lock:
            _log_buffer.append(message)


class _StdStreamProxy:
    """Mirror writes to stdout/stderr while recording them in the shared buffer."""

    def __init__(self, original, stream_name: str) -> None:
        self._original = original
        self._stream_name = stream_name

    def write(self, text: str) -> int:
        # 尽量使用原编码写入，遇到无法编码字符时降级替换，避免 UnicodeEncodeError
        try:
            written = self._original.write(text)
        except UnicodeEncodeError:
            try:
                enc = getattr(self._original, "encoding", None) or "utf-8"
                if hasattr(self._original, "buffer") and self._original.buffer is not None:
                    self._original.buffer.write(text.encode(enc, errors="replace"))
                    written = len(text)
                else:
                    # 回退到替换后再写
                    safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
                    written = self._original.write(safe)
            except Exception:
                # 最后回退：丢弃不可写字符
                safe = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
                written = self._original.write(safe)
        if text and not text.isspace():
            timestamp = datetime.now().strftime(_DATE_FORMAT)
            lines = text.rstrip().splitlines()
            payloads = [f"{timestamp} | {self._stream_name} | {line}" for line in lines if line]
            if payloads:
                with _buffer_lock:
                    _log_buffer.extend(payloads)
        return written

    def flush(self) -> None:
        self._original.flush()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    def fileno(self):
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()


_logging_configured = False


def _configure_logging() -> None:
    global _logging_configured
    if _logging_configured:
        return

    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)
    stream_handler = logging.StreamHandler(sys.__stderr__)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(_RESOLVED_LEVEL)

    buffer_handler = _BufferingHandler()

    # Ensure every record has run_id
    run_filter = RunIdFilter()

    root_logger = logging.getLogger()
    root_logger.setLevel(_RESOLVED_LEVEL)
    # Attach filter to root and handlers so records always carry run_id
    if run_filter not in root_logger.filters:
        root_logger.addFilter(run_filter)
    stream_handler.addFilter(run_filter)
    buffer_handler.addFilter(run_filter)

    # Avoid attaching duplicate handlers if other configuration already exists.
    if not any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers):
        root_logger.addHandler(stream_handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(_RESOLVED_LEVEL)
            # also ensure existing handlers get the filter
            try:
                handler.addFilter(run_filter)
            except Exception:
                pass

    if not any(isinstance(handler, _BufferingHandler) for handler in root_logger.handlers):
        root_logger.addHandler(buffer_handler)

    # Capture residual stdout/stderr writes from legacy paths.
    if not isinstance(sys.stdout, _StdStreamProxy):
        sys.stdout = _StdStreamProxy(sys.stdout, "STDOUT")  # type: ignore[assignment]
    if not isinstance(sys.stderr, _StdStreamProxy):
        sys.stderr = _StdStreamProxy(sys.stderr, "STDERR")  # type: ignore[assignment]

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the unified ArcticRoute format."""
    _configure_logging()
    logger = logging.getLogger(name)
    return logger


def get_recent_output(limit: int = 200) -> List[str]:
    """Return the most recent log/stdout/stderr lines up to ``limit`` entries."""
    if limit <= 0:
        return []
    with _buffer_lock:
        return list(_log_buffer)[-limit:]


def export_recent_output(limit: int = 200) -> str:
    """Render recent output lines as a single newline-delimited string."""
    return "\n".join(get_recent_output(limit))


def iter_output(limit: int = 200) -> Iterable[str]:
    """Yield recent output lines without building an intermediate list."""
    for line in get_recent_output(limit):
        yield line


def configure_logging(structured: bool = True) -> None:
    """可选入口：配置 root logger 的文件输出到 outputs/pipeline_runs.log。

    - structured=True: "%(asctime)s | %(levelname)s | ArcticRoute.%(module)s | %(message)s"
    - structured=False: "%(message)s"

    不改变已有的缓冲/控制台配置，仅追加/更新文件 handler。
    """
    _configure_logging()  # 确保基础流和缓冲已经装好

    fmt = ("%(asctime)s | %(levelname)s | ArcticRoute.%(module)s | %(message)s") if structured else ("%(message)s")
    datefmt = _DATE_FORMAT

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    outputs_dir = os.path.join(os.getcwd(), "outputs")
    try:
        os.makedirs(outputs_dir, exist_ok=True)
    except Exception:
        pass
    log_path = os.path.join(outputs_dir, "pipeline_runs.log")

    file_handler_exists = False
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            # 将已有文件 handler 调整为最新 formatter
            h.setFormatter(formatter)
            file_handler_exists = True
    if not file_handler_exists:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(_RESOLVED_LEVEL)
        # 保证 run_id 注入
        try:
            fh.addFilter(RunIdFilter())
        except Exception:
            pass
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)


__all__ = ["get_logger", "get_recent_output", "export_recent_output", "iter_output", "configure_logging"]
