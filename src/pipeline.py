"""语音转写流水线：SenseVoice 转写。"""

from __future__ import annotations

import logging

from zimu.models import SUPPORTED_AUDIO_EXTENSIONS, PipelineConfig, PipelineResult
from zimu.segment import split_subtitle_segments
from zimu.transcribe import TranscriptionBackendProtocol, create_transcription_service

logger = logging.getLogger(__name__)


class SubtitlePipeline:
    """
    语音转写流水线。

    步骤:
        1. 校验输入音频格式
        2. SenseVoice 转写，生成带时间戳的字幕片段
    """

    def __init__(
        self,
        transcription: TranscriptionBackendProtocol | None = None,
    ) -> None:
        self._transcription = transcription

    def run(self, config: PipelineConfig) -> PipelineResult:
        """
        执行完整转写流程。

        Args:
            config: 流水线配置（含模型路径、设备等）。

        Returns:
            包含转写文本与元信息的 PipelineResult。

        Raises:
            FileNotFoundError: 输入音频或本地模型不存在。
            ValueError: 输入格式不支持或转写结果异常。
        """
        self._validate_input(config)

        transcription = self._transcription or create_transcription_service(config)
        segments, info = transcription.transcribe(
            config.input_audio,
            language=config.language,
        )
        segments = split_subtitle_segments(
            segments,
            max_chars=config.max_subtitle_chars,
            max_duration_sec=config.max_subtitle_duration_sec,
        )

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
        """校验输入文件存在且为支持的音频格式。"""
        if not config.input_audio.is_file():
            raise FileNotFoundError(f"输入音频不存在: {config.input_audio}")

        suffix = config.input_audio.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise ValueError(
                f"不支持的音频格式: {suffix}\n"
                f"支持: {supported}"
            )

    @staticmethod
    def _log_result(result: PipelineResult) -> None:
        """输出 PipelineResult 中的全部信息。"""
        config = result.config
        logger.info("流水线完成:")
        logger.info("  [配置]")
        logger.info("    输入音频: %s", config.input_audio)
        logger.info("    SenseVoice 模型: %s", config.sensevoice_model_path)
        logger.info("    VAD 模型: %s", config.sensevoice_vad_path)
        logger.info("    语言: %s", config.language or "auto")
        logger.info("    推理设备: %s", config.device)
        logger.info("    单条最大字数: %d", config.max_subtitle_chars)
        logger.info("    单条最大时长: %.1fs", config.max_subtitle_duration_sec)
        logger.info("  [转写结果]")
        logger.info("    检测语言: %s", result.detected_language or "未知")
        if result.language_probability is not None:
            logger.info("    语言置信度: %.2f", result.language_probability)
        logger.info("    字幕条数: %d", len(result.segments))
        logger.info("    转写文本: %s", result.transcription_text)
        if result.segments:
            logger.info("  [字幕片段]")
            for segment in result.segments:
                logger.info(
                    "    #%d [%.3f -> %.3f] %s",
                    segment.index,
                    segment.start,
                    segment.end,
                    segment.text,
                )
