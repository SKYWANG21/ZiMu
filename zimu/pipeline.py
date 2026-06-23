"""字幕生成流水线：串联音频提取、转写、SRT 写入与硬字幕烧录。"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from zimu.ffmpeg import FFmpegService
from zimu.models import PipelineConfig, PipelineResult, SubtitleSegment
from zimu.srt import SrtWriter
from zimu.transcribe import TranscriptionService

logger = logging.getLogger(__name__)


class SubtitlePipeline:
    """
    视频字幕生成流水线。

    步骤:
        1. ffmpeg 从 MP4 提取 WAV
        2. faster-whisper 转写生成字幕片段
        3. 写入 SRT 文件
        4. ffmpeg 将 SRT 硬烧录为带字幕 MP4
    """

    def __init__(
        self,
        ffmpeg: FFmpegService | None = None,
        transcription: TranscriptionService | None = None,
    ) -> None:
        self._ffmpeg = ffmpeg or FFmpegService()
        self._transcription = transcription

    def run(self, config: PipelineConfig) -> PipelineResult:
        """
        执行完整字幕生成流程。

        Args:
            config: 流水线配置。

        Returns:
            包含输出路径与转写信息的 PipelineResult。

        Raises:
            FileNotFoundError: 输入视频不存在。
            FFmpegError: ffmpeg 命令失败。
        """
        self._validate_input(config)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        transcription = self._transcription or TranscriptionService(
            model_path=config.model_path,
            device=config.device,
        )

        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        wav_path: Path

        if config.keep_temp:
            wav_path = config.output_dir / f"{config.stem}_audio.wav"
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="zimu_")
            wav_path = Path(temp_dir.name) / "audio.wav"

        try:
            self._ffmpeg.extract_audio(config.input_video, wav_path)
            segments, info = transcription.transcribe(
                wav_path,
                language=config.language,
            )
            SrtWriter.write(segments, config.srt_path)
            self._ffmpeg.burn_subtitles(
                config.input_video,
                config.srt_path,
                config.subtitled_video_path,
                font_name=config.subtitle_font,
                font_size=config.subtitle_font_size,
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
            elif not config.keep_temp and wav_path.exists():
                wav_path.unlink(missing_ok=True)

        result = PipelineResult(
            config=config,
            segments=segments,
            detected_language=info.language,
            language_probability=info.language_probability,
        )
        self._log_result(result)
        return result

    @staticmethod
    def _validate_input(config: PipelineConfig) -> None:
        """校验输入文件存在且格式为 MP4。"""
        if not config.input_video.is_file():
            raise FileNotFoundError(f"输入视频不存在: {config.input_video}")
        if config.input_video.suffix.lower() != ".mp4":
            raise ValueError(f"仅支持 .mp4 输入，当前: {config.input_video.suffix}")

    @staticmethod
    def _log_result(result: PipelineResult) -> None:
        """输出流水线完成摘要。"""
        logger.info("流水线完成:")
        logger.info("  SRT: %s", result.srt_path)
        logger.info("  视频: %s", result.subtitled_video_path)
        logger.info("  字幕条数: %d", len(result.segments))
