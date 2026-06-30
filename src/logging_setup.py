"""错误时持久化全链路日志到 log/ 目录。"""

from __future__ import annotations

import logging
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterator

DEFAULT_LOG_DIR = Path("log")


class _BufferingHandler(logging.Handler):
    """将日志记录缓存在内存中，供异常时写入文件。"""

    def __init__(self) -> None:
        super().__init__()
        self._buffer = StringIO()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self._buffer.write(msg + "\n")

    def getvalue(self) -> str:
        return self._buffer.getvalue()


def _persist_logs(handler: _BufferingHandler, log_dir: Path, exc: BaseException) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_error.log"

    content = handler.getvalue()
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_path.write_text(
        f"{content}\n--- exception ---\n{tb}",
        encoding="utf-8",
    )
    print(f"错误日志已写入: {log_path.resolve()}", file=sys.stderr)
    return log_path


@contextmanager
def error_log_session(log_dir: Path = DEFAULT_LOG_DIR) -> Iterator[None]:
    """
    运行期间缓冲全链路日志；成功时不写文件，异常时写入 log/ 目录。

    Args:
        log_dir: 错误日志输出目录，默认项目根目录下 log/。
    """
    root = logging.getLogger()
    handler = _BufferingHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    try:
        yield
    except BaseException as exc:
        _persist_logs(handler, log_dir, exc)
        raise
    finally:
        root.removeHandler(handler)
