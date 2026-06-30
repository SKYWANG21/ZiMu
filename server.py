"""ZiMu HTTP 服务入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from zimu.api import create_app
from zimu.settings import DEFAULT_CONFIG_PATH, load_settings

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建服务启动参数解析器。"""
    parser = argparse.ArgumentParser(description="ZiMu 语音转写 HTTP 服务。")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="监听地址（默认读取 config.json 中 server.host）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="监听端口（默认读取 config.json 中 server.port）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出调试日志",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    """配置根日志级别与格式。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[list[str]] = None) -> int:
    """启动 uvicorn 服务。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    host = args.host if args.host is not None else settings.server_host
    port = args.port if args.port is not None else settings.server_port

    app = create_app(settings)
    logger.info("启动 HTTP 服务: http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
