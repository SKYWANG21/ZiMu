"""ZiMu — 基于 faster-whisper 与 ffmpeg 的视频字幕生成工具包。"""

from zimu.models import DEFAULT_MODEL_PATH, PipelineConfig, PipelineResult, SubtitleSegment
from zimu.pipeline import SubtitlePipeline

__all__ = [
    "DEFAULT_MODEL_PATH",
    "PipelineConfig",
    "PipelineResult",
    "SubtitlePipeline",
    "SubtitleSegment",
]
