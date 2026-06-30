"""ZiMu — 基于 faster-whisper / SenseVoice / 远程 API 与 ffmpeg 的视频字幕生成工具包。"""

from zimu.models import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REMOTE_API_URL,
    DEFAULT_SENSEVOICE_MODEL_PATH,
    DEFAULT_SENSEVOICE_VAD_PATH,
    PipelineConfig,
    PipelineResult,
    SubtitleSegment,
    TranscriptionBackend,
)
from zimu.pipeline import SubtitlePipeline
from zimu.transcribe import (
    RemoteTranscriptionService,
    SenseVoiceTranscriptionService,
    TranscriptionService,
    WhisperTranscriptionService,
    create_transcription_service,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_REMOTE_API_URL",
    "DEFAULT_SENSEVOICE_MODEL_PATH",
    "DEFAULT_SENSEVOICE_VAD_PATH",
    "PipelineConfig",
    "PipelineResult",
    "RemoteTranscriptionService",
    "SenseVoiceTranscriptionService",
    "SubtitlePipeline",
    "SubtitleSegment",
    "TranscriptionBackend",
    "TranscriptionService",
    "WhisperTranscriptionService",
    "create_transcription_service",
]
