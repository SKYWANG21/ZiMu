"""SRT 字幕文件的生成与写入。"""

from __future__ import annotations

from pathlib import Path

from faster_whisper.utils import format_timestamp

from src.models import SubtitleSegment


class SrtWriter:
    """将字幕片段序列化为标准 SRT 格式并写入文件。"""

    @staticmethod
    def format_time(seconds: float) -> str:
        """将秒数格式化为 SRT 时间戳（HH:MM:SS,mmm）。"""
        return format_timestamp(seconds, always_include_hours=True, decimal_marker=",")

    @staticmethod
    def normalize_text(text: str) -> str:
        """清理转写文本：去除首尾空白并折叠多余空行。"""
        lines = [line.strip() for line in text.strip().splitlines()]
        return "\n".join(line for line in lines if line)

    @classmethod
    def segment_to_block(cls, segment: SubtitleSegment) -> str:
        """将单条字幕片段格式化为 SRT 文本块。"""
        start = cls.format_time(segment.start)
        end = cls.format_time(segment.end)
        text = cls.normalize_text(segment.text)
        return f"{segment.index}\n{start} --> {end}\n{text}"

    @classmethod
    def segments_to_content(cls, segments: list[SubtitleSegment]) -> str:
        """将全部片段拼接为完整 SRT 文档内容。"""
        blocks = [cls.segment_to_block(seg) for seg in segments]
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    @classmethod
    def write(cls, segments: list[SubtitleSegment], output_path: Path) -> Path:
        """
        将字幕片段写入 SRT 文件。

        Args:
            segments: 有序字幕片段列表。
            output_path: 目标 .srt 路径。

        Returns:
            写入后的文件路径。
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = cls.segments_to_content(segments)
        output_path.write_text(content, encoding="utf-8")
        return output_path
