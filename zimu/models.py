"""领域模型与配置类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

# 默认本地 Whisper 模型目录（CTranslate2 转换后的 faster-whisper 模型）
DEFAULT_MODEL_PATH = Path(r"D:\models\faster-whisper-medium")


@dataclass(frozen=True)
class SubtitleSegment:
    """单条字幕片段，对应 SRT 中的一条记录。"""

    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class PipelineConfig:
    """字幕生成流水线的运行配置。"""

    input_video: Path
    output_dir: Path
    model_path: Path = DEFAULT_MODEL_PATH
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
