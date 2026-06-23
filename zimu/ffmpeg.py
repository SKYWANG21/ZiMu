"""ffmpeg 子进程封装：音频提取与硬字幕烧录。"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Whisper 推荐 16 kHz 单声道 PCM
_WHISPER_SAMPLE_RATE = 16_000

# 默认 ffmpeg 绝对路径（venv 隔离环境可能读不到系统 PATH）
DEFAULT_FFMPEG_EXECUTABLE = Path(r"D:\ffmpeg\bin\ffmpeg.exe")


class FFmpegError(RuntimeError):
    """ffmpeg 命令执行失败时抛出。"""


class FFmpegService:
    """封装本地 ffmpeg CLI 的常用操作。"""

    def __init__(self, executable: str | Path | None = None) -> None:
        self._executable = Path(executable or DEFAULT_FFMPEG_EXECUTABLE)
        self._ensure_available()

    @property
    def executable(self) -> Path:
        return self._executable

    def _ensure_available(self) -> None:
        """确认 ffmpeg 可执行文件存在且可运行。"""
        if not self._executable.is_file():
            raise FFmpegError(
                f"未找到 ffmpeg 可执行文件: {self._executable}\n"
                "请确认已安装 ffmpeg，或构造 FFmpegService 时传入正确路径。"
            )
        self._run([str(self._executable), "-version"], capture_output=True)

    def extract_audio(self, video_path: Path, output_wav: Path) -> Path:
        """
        从视频中提取 16 kHz 单声道 WAV，供 Whisper 转写使用。

        Args:
            video_path: 输入 MP4 路径。
            output_wav: 输出 WAV 路径。

        Returns:
            输出 WAV 的路径。
        """
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self._executable),
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(_WHISPER_SAMPLE_RATE),
            "-ac",
            "1",
            str(output_wav),
        ]
        logger.info("提取音频: %s -> %s", video_path.name, output_wav.name)
        self._run(cmd)
        return output_wav

    def burn_subtitles(
        self,
        video_path: Path,
        srt_path: Path,
        output_path: Path,
        *,
        font_name: str = "Microsoft YaHei",
        font_size: int = 24,
    ) -> Path:
        """
        使用 subtitles 滤镜将 SRT 硬烧录到视频画面。

        Args:
            video_path: 输入 MP4 路径。
            srt_path: SRT 字幕文件路径。
            output_path: 输出 MP4 路径。
            font_name: 字幕字体（需支持目标语言字符集）。
            font_size: 字幕字号。

        Returns:
            输出视频路径。
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        escaped_srt = self._escape_subtitles_filter_path(srt_path)
        force_style = (
            f"FontName={font_name},"
            f"FontSize={font_size},"
            "PrimaryColour=&HFFFFFF&,"
            "OutlineColour=&H000000&,"
            "Outline=2"
        )
        vf = f"subtitles='{escaped_srt}':force_style='{force_style}'"
        cmd = [
            str(self._executable),
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-c:a",
            "copy",
            str(output_path),
        ]
        logger.info("烧录字幕: %s + %s -> %s", video_path.name, srt_path.name, output_path.name)
        self._run(cmd)
        return output_path

    @staticmethod
    def _escape_subtitles_filter_path(path: Path) -> str:
        """
        转义 Windows 路径，供 ffmpeg subtitles 滤镜使用。

        ffmpeg 滤镜解析器对冒号、反斜杠敏感，需统一为正斜杠并转义特殊字符。
        """
        normalized = path.resolve().as_posix()
        # 盘符冒号（如 D:）在滤镜中是特殊字符
        if len(normalized) >= 2 and normalized[1] == ":":
            normalized = normalized[0] + "\\:" + normalized[2:]
        return normalized.replace("'", r"\'")

    def _run(
        self,
        cmd: list[str],
        *,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """执行 ffmpeg 命令，失败时抛出 FFmpegError。"""
        logger.debug("执行命令: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=capture_output,
                text=True,
            )
            return result
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or exc.stdout or "(无输出)"
            raise FFmpegError(f"ffmpeg 执行失败 (exit {exc.returncode}):\n{stderr}") from exc
