"""领域模型与配置类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# 默认本地模型路径
# ---------------------------------------------------------------------------

# faster-whisper（CTranslate2 转换后的 Whisper 权重）
DEFAULT_MODEL_PATH = Path(r"D:\models\faster-whisper-large-v3")

# SenseVoice 主模型（FunASR AutoModel 本地目录，需用户手动下载）
DEFAULT_SENSEVOICE_MODEL_PATH = Path(r"D:\models\SenseVoiceSmall")

# FSMN-VAD 模型（SenseVoice 分段与时间戳必需，需用户手动下载）
DEFAULT_SENSEVOICE_VAD_PATH = Path(r"D:\models\speech_fsmn_vad_zh-cn-16k")

# 转写后端类型：whisper 为默认，sensevoice 为可选
TranscriptionBackend = Literal["whisper", "sensevoice"]


@dataclass(frozen=True)
class SubtitleSegment:
    """单条字幕片段，对应 SRT 中的一条记录。"""

    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class PipelineConfig:
    """
    字幕生成流水线的运行配置。

    通过 ``backend`` 选择转写引擎：
    - ``whisper``：使用 ``model_path`` 指定的 faster-whisper 本地模型
    - ``sensevoice``：使用 ``sensevoice_model_path`` 与 ``sensevoice_vad_path``
    """

    input_video: Path
    output_dir: Path
    # Whisper 后端专用：本地 faster-whisper 模型目录
    model_path: Path = DEFAULT_MODEL_PATH
    # 转写后端选择，默认 whisper 以保持向后兼容
    backend: TranscriptionBackend = "whisper"
    # SenseVoice 后端专用：主模型与 VAD 本地目录
    sensevoice_model_path: Path = DEFAULT_SENSEVOICE_MODEL_PATH
    sensevoice_vad_path: Path = DEFAULT_SENSEVOICE_VAD_PATH
    # 两种后端共用：None 表示自动检测 / auto
    language: Optional[str] = None
    device: Literal["auto", "cuda", "cpu"] = "auto"
    keep_temp: bool = False
    # 烧录字幕时的样式（ASS force_style 语法）
    subtitle_font: str = "Microsoft YaHei"
    subtitle_font_size: int = 24

    @property
    def stem(self) -> str:
        """输入视频的文件名（不含扩展名）。"""
        return self.input_video.stem

    @property
    def srt_path(self) -> Path:
        """SRT 字幕输出路径。"""
        return self.output_dir / f"{self.stem}.srt"

    @property
    def subtitled_video_path(self) -> Path:
        """硬字幕视频输出路径。"""
        return self.output_dir / f"{self.stem}_subtitled.mp4"


@dataclass
class PipelineResult:
    """流水线执行结果。"""

    config: PipelineConfig
    segments: list[SubtitleSegment] = field(default_factory=list)
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None

    @property
    def srt_path(self) -> Path:
        return self.config.srt_path

    @property
    def subtitled_video_path(self) -> Path:
        return self.config.subtitled_video_path
