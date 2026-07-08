"""src CLI — 视频字幕生成 Demo 入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

from src.models import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REMOTE_API_URL,
    DEFAULT_SENSEVOICE_MODEL_PATH,
    DEFAULT_SENSEVOICE_VAD_PATH,
    PipelineConfig,
    TranscriptionBackend,
)
from src.pipeline import SubtitlePipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "基于 faster-whisper / SenseVoice / 远程 API + ffmpeg，为 MP4 视频生成硬字幕。"
        ),
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

    # --- 转写后端与模型路径 ---
    parser.add_argument(
        "--backend",
        choices=("whisper", "sensevoice", "remote"),
        default="whisper",
        help="转写后端：whisper（默认）、sensevoice 或 remote",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            f"本地 Whisper 模型目录，仅 backend=whisper 时生效"
            f"（默认: {DEFAULT_MODEL_PATH}）"
        ),
    )
    parser.add_argument(
        "--sensevoice-model-path",
        type=Path,
        default=DEFAULT_SENSEVOICE_MODEL_PATH,
        help=(
            f"SenseVoice 本地模型目录，仅 backend=sensevoice 时生效"
            f"（默认: {DEFAULT_SENSEVOICE_MODEL_PATH}）"
        ),
    )
    parser.add_argument(
        "--sensevoice-vad-path",
        type=Path,
        default=DEFAULT_SENSEVOICE_VAD_PATH,
        help=(
            f"FSMN-VAD 本地模型目录，SenseVoice 分段与时间戳必需"
            f"（默认: {DEFAULT_SENSEVOICE_VAD_PATH}）"
        ),
    )
    parser.add_argument(
        "--remote-api-url",
        default=DEFAULT_REMOTE_API_URL,
        help=(
            f"远程转写 API 地址，仅 backend=remote 时生效"
            f"（默认: {DEFAULT_REMOTE_API_URL}）"
        ),
    )

    # --- 两种后端共用参数 ---
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

    # --- 字幕烧录样式 ---
    parser.add_argument(
        "--font",
        default="Microsoft YaHei",
        help="烧录字幕字体（默认: Microsoft YaHei）",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="烧录字幕字号（默认: 12）",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=14,
        help="单条字幕最大字数（默认: 14）",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=3.5,
        help="单条字幕最大时长，单位秒（默认: 3.5）",
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
        args.output_dir.resolve() if args.output_dir is not None else input_video.parent
    )
    device: Literal["auto", "cuda", "cpu"] = args.device
    backend: TranscriptionBackend = args.backend

    return PipelineConfig(
        input_video=input_video,
        output_dir=output_dir,
        backend=backend,
        model_path=args.model_path.resolve(),
        sensevoice_model_path=args.sensevoice_model_path.resolve(),
        sensevoice_vad_path=args.sensevoice_vad_path.resolve(),
        remote_api_url=args.remote_api_url,
        language=args.language,
        device=device,
        keep_temp=args.keep_temp,
        subtitle_font=args.font,
        subtitle_font_size=args.font_size,
        max_subtitle_chars=args.max_chars,
        max_subtitle_duration_sec=args.max_duration,
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
        if result.language_probability is not None:
            prob = result.language_probability
            print(f"语言:  {result.detected_language} ({prob:.0%})")
        else:
            print(f"语言:  {result.detected_language}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
