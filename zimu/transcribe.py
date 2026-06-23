"""faster-whisper 语音转写封装。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment as WhisperSegment

from zimu.models import DEFAULT_MODEL_PATH, SubtitleSegment

logger = logging.getLogger(__name__)

DeviceChoice = Literal["auto", "cuda", "cpu"]


@dataclass(frozen=True)
class TranscriptionInfo:
    """转写元信息（语言检测结果等）。"""

    language: Optional[str]
    language_probability: Optional[float]
    duration: float


class TranscriptionService:
    """封装 WhisperModel 的加载与音频转写逻辑。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: DeviceChoice = "auto",
    ) -> None:
        self._model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._device = device
        self._model: Optional[WhisperModel] = None

    @property
    def model(self) -> WhisperModel:
        """懒加载 Whisper 模型，避免未使用时占用内存。"""
        if self._model is None:
            self._ensure_model_available()
            resolved_device, compute_type = self._resolve_device_and_compute_type()
            logger.info(
                "加载本地模型: path=%s device=%s compute_type=%s",
                self._model_path,
                resolved_device,
                compute_type,
            )
            self._model = WhisperModel(
                str(self._model_path),
                device=resolved_device,
                compute_type=compute_type,
                local_files_only=True,
            )
        return self._model

    def _ensure_model_available(self) -> None:
        """确认本地模型目录存在且包含必要文件。"""
        if not self._model_path.is_dir():
            raise FileNotFoundError(
                f"本地模型目录不存在: {self._model_path}\n"
                "请先将 faster-whisper 模型下载到该目录。"
            )
        required_files = ("config.json", "model.bin")
        missing = [name for name in required_files if not (self._model_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"本地模型目录不完整: {self._model_path}\n"
                f"缺少文件: {', '.join(missing)}"
            )

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
    ) -> tuple[list[SubtitleSegment], TranscriptionInfo]:
        """
        转写音频文件并返回带时间戳的字幕片段。

        Args:
            audio_path: 16 kHz WAV 路径。
            language: 指定语言代码（如 zh、en）；None 表示自动检测。

        Returns:
            (字幕片段列表, 转写元信息) 元组。
        """
        logger.info("开始转写: %s", audio_path.name)
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
            beam_size=5,
        )
        whisper_segments = list(segments_iter)
        subtitle_segments = self._to_subtitle_segments(whisper_segments)
        transcription_info = TranscriptionInfo(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )
        logger.info(
            "转写完成: %d 条字幕, language=%s (%.2f), duration=%.1fs",
            len(subtitle_segments),
            transcription_info.language,
            transcription_info.language_probability or 0.0,
            transcription_info.duration,
        )
        return subtitle_segments, transcription_info

    def _resolve_device_and_compute_type(self) -> tuple[str, str]:
        """根据配置解析实际 device 与 compute_type。"""
        if self._device == "cpu":
            return "cpu", "int8"
        if self._device == "cuda":
            return "cuda", "float16"

        # auto: 尝试 CUDA，不可用时回退 CPU
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
        return "cpu", "int8"

    @staticmethod
    def _to_subtitle_segments(
        whisper_segments: list[WhisperSegment],
    ) -> list[SubtitleSegment]:
        """将 faster-whisper Segment 转换为领域模型。"""
        return [
            SubtitleSegment(
                index=index,
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
            for index, segment in enumerate(whisper_segments, start=1)
        ]
