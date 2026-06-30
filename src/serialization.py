"""PipelineResult JSON 序列化。"""

from __future__ import annotations

from typing import Any, Optional

from zimu.models import PipelineConfig, PipelineResult, SubtitleSegment


def segment_to_dict(segment: SubtitleSegment) -> dict[str, Any]:
    """将单条字幕片段转为 JSON 可序列化字典。"""
    return {
        "index": segment.index,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }


def pipeline_result_to_dict(
    result: PipelineResult,
    *,
    input_filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    将 PipelineResult 转为 JSON 可序列化字典。

    Args:
        result: 流水线执行结果。
        input_filename: 对外展示的文件名；不传则使用 config 中的输入文件名。
    """
    filename = input_filename or result.config.input_audio.name
    return {
        "transcription_text": result.transcription_text,
        "detected_language": result.detected_language,
        "language_probability": result.language_probability,
        "segments": [segment_to_dict(s) for s in result.segments],
        "config": _config_to_dict(result.config, input_filename=filename),
    }


def _config_to_dict(
    config: PipelineConfig,
    *,
    input_filename: str,
) -> dict[str, Any]:
    return {
        "input_filename": input_filename,
        "language": config.language,
        "device": config.device,
        "max_subtitle_chars": config.max_subtitle_chars,
        "max_subtitle_duration_sec": config.max_subtitle_duration_sec,
        "sensevoice_model_path": str(config.sensevoice_model_path),
        "sensevoice_vad_path": str(config.sensevoice_vad_path),
    }
