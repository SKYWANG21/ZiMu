"""从 config.json 加载应用配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from zimu.models import (
    DEFAULT_SENSEVOICE_MODEL_PATH,
    DEFAULT_SENSEVOICE_VAD_PATH,
    PipelineConfig,
)

DeviceChoice = Literal["auto", "cuda", "cpu"]
_VALID_DEVICES: frozenset[str] = frozenset({"auto", "cuda", "cpu"})

DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass(frozen=True)
class AppSettings:
    """应用级配置（来自 config.json）。"""

    device: DeviceChoice = "cpu"
    max_subtitle_chars: int = 14
    max_subtitle_duration_sec: float = 3.5
    sensevoice_model_path: Path = DEFAULT_SENSEVOICE_MODEL_PATH
    sensevoice_vad_path: Path = DEFAULT_SENSEVOICE_VAD_PATH
    server_host: str = "0.0.0.0"
    server_port: int = 8000


def _default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


def load_settings(config_path: Path | None = None) -> AppSettings:
    """
    从 config.json 加载配置，缺失字段使用代码默认值。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: JSON 格式或字段值非法。
    """
    path = config_path or _default_config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在: {path.resolve()}\n"
            f"请复制 config.example.json 为 config.json 并编辑后重试。"
        )

    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置文件 JSON 解析失败: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"配置文件根节点必须为对象: {path}")

    server = raw.get("server", {})
    if server is not None and not isinstance(server, dict):
        raise ValueError("配置项 server 必须为对象")

    device = _parse_device(raw.get("device", "cpu"))
    max_chars = _parse_positive_int(
        raw.get("max_subtitle_chars", 14),
        field="max_subtitle_chars",
    )
    max_duration = _parse_positive_float(
        raw.get("max_subtitle_duration_sec", 3.5),
        field="max_subtitle_duration_sec",
    )
    model_path = _parse_path(
        raw.get("sensevoice_model_path"),
        default=DEFAULT_SENSEVOICE_MODEL_PATH,
        field="sensevoice_model_path",
    )
    vad_path = _parse_path(
        raw.get("sensevoice_vad_path"),
        default=DEFAULT_SENSEVOICE_VAD_PATH,
        field="sensevoice_vad_path",
    )
    server_host = _parse_str(server.get("host", "0.0.0.0"), field="server.host")
    server_port = _parse_positive_int(server.get("port", 8000), field="server.port")

    return AppSettings(
        device=device,
        max_subtitle_chars=max_chars,
        max_subtitle_duration_sec=max_duration,
        sensevoice_model_path=model_path,
        sensevoice_vad_path=vad_path,
        server_host=server_host,
        server_port=server_port,
    )


def to_pipeline_config(
    settings: AppSettings,
    input_audio: Path,
    language: Optional[str] = None,
) -> PipelineConfig:
    """将 AppSettings 与运行时参数合并为 PipelineConfig。"""
    return PipelineConfig(
        input_audio=input_audio.resolve(),
        sensevoice_model_path=settings.sensevoice_model_path.resolve(),
        sensevoice_vad_path=settings.sensevoice_vad_path.resolve(),
        language=language,
        device=settings.device,
        max_subtitle_chars=settings.max_subtitle_chars,
        max_subtitle_duration_sec=settings.max_subtitle_duration_sec,
    )


def _parse_device(value: Any) -> DeviceChoice:
    if not isinstance(value, str) or value not in _VALID_DEVICES:
        allowed = ", ".join(sorted(_VALID_DEVICES))
        raise ValueError(f"device 必须为 {allowed} 之一，当前值: {value!r}")
    return value  # type: ignore[return-value]


def _parse_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} 必须为正整数，当前值: {value!r}")
    if value <= 0:
        raise ValueError(f"{field} 必须为正整数，当前值: {value}")
    return value


def _parse_positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须为正数，当前值: {value!r}")
    result = float(value)
    if result <= 0:
        raise ValueError(f"{field} 必须为正数，当前值: {value}")
    return result


def _parse_str(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须为非空字符串，当前值: {value!r}")
    return value.strip()


def _parse_path(value: Any, *, default: Path, field: str) -> Path:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须为非空路径字符串，当前值: {value!r}")
    return Path(value.strip())
