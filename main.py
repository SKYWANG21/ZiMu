"""ZiMu CLI — 音频语音转写 Demo 入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from zimu.logging_setup import error_log_session
from zimu.models import SUPPORTED_AUDIO_EXTENSIONS
from zimu.pipeline import SubtitlePipeline
from zimu.settings import DEFAULT_CONFIG_PATH, load_settings, to_pipeline_config

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
    parser = argparse.ArgumentParser(
        description="基于 SenseVoice 的音频语音转写，输出转写文本。",
    )
    parser.add_argument(
        "input",
        type=Path,
        help=f"输入音频路径（支持: {supported}）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="指定语言代码，如 zh、en（默认自动检测）",
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
    """
    CLI 主函数。

    Returns:
        进程退出码，0 表示成功。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        with error_log_session():
            settings = load_settings(args.config)
            config = to_pipeline_config(
                settings,
                input_audio=args.input,
                language=args.language,
            )
            pipeline = SubtitlePipeline()
            result = pipeline.run(config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("处理失败: %s", exc)
        return 1

    print(result.transcription_text)
    if result.detected_language:
        if result.language_probability is not None:
            prob = result.language_probability
            logger.info("语言: %s (%.0f%%)", result.detected_language, prob * 100)
        else:
            logger.info("语言: %s", result.detected_language)
    return 0


if __name__ == "__main__":
    sys.exit(main())
