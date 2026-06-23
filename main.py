"""ZiMu CLI — 视频字幕生成 Demo 入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

from zimu.models import DEFAULT_MODEL_PATH, PipelineConfig
from zimu.pipeline import SubtitlePipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="基于 faster-whisper + ffmpeg 为 MP4 视频生成硬字幕。",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="输入 .mp4 视频路径",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认与输入文件同目录）",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"本地 Whisper 模型目录（默认: {DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="指定语言代码，如 zh、en（默认自动检测）",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="推理设备（默认: auto）",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留中间提取的 WAV 音频文件",
    )
    parser.add_argument(
        "--font",
        default="Microsoft YaHei",
        help="烧录字幕字体（默认: Microsoft YaHei）",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=24,
        help="烧录字幕字号（默认: 24）",
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


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """将 CLI 参数转换为 PipelineConfig。"""
    input_video: Path = args.input.resolve()
    output_dir: Path = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_video.parent
    )
    device: Literal["auto", "cuda", "cpu"] = args.device
    return PipelineConfig(
        input_video=input_video,
        output_dir=output_dir,
        model_path=args.model_path.resolve(),
        language=args.language,
        device=device,
        keep_temp=args.keep_temp,
        subtitle_font=args.font,
        subtitle_font_size=args.font_size,
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
        config = build_config(args)
        pipeline = SubtitlePipeline()
        result = pipeline.run(config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("处理失败: %s", exc)
        return 1

    print(f"SRT:   {result.srt_path}")
    print(f"视频:  {result.subtitled_video_path}")
    if result.detected_language:
        prob = result.language_probability or 0.0
        print(f"语言:  {result.detected_language} ({prob:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
