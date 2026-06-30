"""领域模型与配置类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# 默认本地模型路径
# ---------------------------------------------------------------------------

# SenseVoice 主模型（FunASR AutoModel 本地目录，需用户手动下载）
DEFAULT_SENSEVOICE_MODEL_PATH = Path(r"D:\models\SenseVoiceSmall")

# FSMN-VAD 模型（SenseVoice 分段与时间戳必需，需用户手动下载）
DEFAULT_SENSEVOICE_VAD_PATH = Path(r"D:\models\speech_fsmn_vad_zh-cn-16k")

# 支持的输入音频扩展名
SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
)


@dataclass(frozen=True)
class SubtitleSegment:
    """单条转写片段（含时间戳与文本）。"""

    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class PipelineConfig:
    """语音转写流水线的运行配置。"""

    input_audio: Path
    sensevoice_model_path: Path = DEFAULT_SENSEVOICE_MODEL_PATH
    sensevoice_vad_path: Path = DEFAULT_SENSEVOICE_VAD_PATH
    language: Optional[str] = None
    device: Literal["auto", "cuda", "cpu"] = "cpu"
    max_subtitle_chars: int = 14
    max_subtitle_duration_sec: float = 3.5

    @property
    def stem(self) -> str:
        """输入音频的文件名（不含扩展名）。"""
        return self.input_audio.stem


@dataclass
class PipelineResult:
    """流水线执行结果。"""

    config: PipelineConfig
    segments: list[SubtitleSegment] = field(default_factory=list)
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None

    @property
    def transcription_text(self) -> str:
        """将全部字幕片段拼接为连续文本。"""
        return "".join(segment.text for segment in self.segments)
