"""语音转写封装：SenseVoice（FunASR）后端。"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Protocol, runtime_checkable

from zimu.models import (
    DEFAULT_SENSEVOICE_MODEL_PATH,
    DEFAULT_SENSEVOICE_VAD_PATH,
    PipelineConfig,
    SubtitleSegment,
)
from zimu.segment import ends_sentence, split_text_chunks

logger = logging.getLogger(__name__)

DeviceChoice = Literal["auto", "cuda", "cpu"]

# SenseVoice 输出文本中的语言标签，如 <|zh|>、|<|en|>
_SENSEVOICE_LANG_TAG = re.compile(r"<\|([a-z]{2,10})\|>")
_SENSEVOICE_LANG_SPLIT = re.compile(r"<\|(zh|en|yue|ja|ko|nospeech)\|>")

# ISO 语言代码 -> SenseVoice language 参数（FunASR 文档约定）
_ISO_TO_SENSEVOICE_LANG: dict[str, str] = {
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "yue": "yue",
    "nospeech": "nospeech",
}


@dataclass(frozen=True)
class TranscriptionInfo:
    """转写元信息（语言检测结果、音频时长等）。"""

    language: Optional[str]
    language_probability: Optional[float]
    duration: float


@runtime_checkable
class TranscriptionBackendProtocol(Protocol):
    """转写后端统一接口，供流水线按配置选择实现。"""

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
    ) -> tuple[list[SubtitleSegment], TranscriptionInfo]:
        """转写音频并返回带时间戳的字幕片段与元信息。"""
        ...


class BaseTranscriptionService(ABC):
    """转写服务抽象基类，定义公共接口与本地模型目录校验逻辑。"""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
    ) -> tuple[list[SubtitleSegment], TranscriptionInfo]: ...

    @staticmethod
    def _ensure_local_model_dir(
        model_path: Path,
        *,
        label: str,
        required_any: tuple[str, ...] = ("model.pt", "model.onnx"),
        required_all: tuple[str, ...] = ("config.yaml",),
    ) -> None:
        """
        校验本地模型目录存在且包含 FunASR 所需的关键文件。

        Args:
            model_path: 模型根目录。
            label: 错误提示中使用的模型名称。
            required_any: 至少存在其中之一的权重文件名。
            required_all: 必须全部存在的配置文件名。
        """
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"{label} 本地模型目录不存在: {model_path}\n"
                "请手动下载模型到该目录后再运行（本项目不会自动在线下载）。"
            )

        missing_required = [
            name for name in required_all if not (model_path / name).is_file()
        ]
        if missing_required:
            raise FileNotFoundError(
                f"{label} 本地模型目录不完整: {model_path}\n"
                f"缺少文件: {', '.join(missing_required)}"
            )

        if not any((model_path / name).is_file() for name in required_any):
            raise FileNotFoundError(
                f"{label} 本地模型目录不完整: {model_path}\n"
                f"缺少权重文件（需其一）: {', '.join(required_any)}"
            )


class SenseVoiceTranscriptionService(BaseTranscriptionService):
    """
    SenseVoice（FunASR）转写后端。

    依赖本地 SenseVoiceSmall 与 FSMN-VAD 模型目录，通过 VAD 分段获取
    ``sentence_info`` 中的毫秒级时间戳，再经 ``rich_transcription_postprocess``
    去除 <|zh|> 等富文本标签后输出文本。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        vad_path: str | Path | None = None,
        device: DeviceChoice = "cpu",
    ) -> None:
        self._model_path = Path(model_path or DEFAULT_SENSEVOICE_MODEL_PATH)
        self._vad_path = Path(vad_path or DEFAULT_SENSEVOICE_VAD_PATH)
        self._device = device
        self._model: object | None = None

    @property
    def model(self) -> object:
        """懒加载 FunASR AutoModel（SenseVoice + VAD），仅使用本地路径。"""
        if self._model is None:
            self._ensure_sensevoice_models_available()
            resolved_device = self._resolve_device()
            logger.info(
                "加载 SenseVoice 模型: model=%s vad=%s device=%s",
                self._model_path,
                self._vad_path,
                resolved_device,
            )
            # 延迟导入：仅在使用 SenseVoice 后端时才需要 funasr / torch
            from funasr import AutoModel

            self._model = AutoModel(
                model=str(self._model_path),
                vad_model=str(self._vad_path),
                vad_kwargs={"max_single_segment_time": 30000},
                device=resolved_device,
                disable_update=True,
            )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
    ) -> tuple[list[SubtitleSegment], TranscriptionInfo]:
        """
        使用 SenseVoice + VAD 转写音频。

        Args:
            audio_path: 输入音频文件路径（如 .wav、.mp3）。
            language: 语言提示；None 映射为 SenseVoice 的 ``auto``。
        """
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        sensevoice_lang = self._map_language(language)
        logger.info(
            "[SenseVoice] 开始转写: %s language=%s",
            audio_path.name,
            sensevoice_lang,
        )

        raw_result = self.model.generate(
            input=str(audio_path),
            cache={},
            language=sensevoice_lang,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
            output_timestamp=True,
        )

        if not raw_result:
            raise ValueError("SenseVoice 未返回转写结果，请检查音频文件与模型目录。")

        result_item = raw_result[0]
        sentence_info = self._resolve_sentence_info(
            result_item,
            audio_path=audio_path,
            postprocess=rich_transcription_postprocess,
        )

        if not sentence_info:
            raise ValueError(
                "SenseVoice 未返回可用的时间戳分段（sentence_info / timestamp）。\n"
                "请确认已安装 funasr>=1.1，且音频含可识别语音。"
            )

        subtitle_segments = self._to_subtitle_segments(
            sentence_info,
            postprocess=rich_transcription_postprocess,
        )
        detected_language = self._detect_language(
            sentence_info,
            fallback=language,
        )
        duration = self._estimate_duration(result_item, subtitle_segments)

        transcription_info = TranscriptionInfo(
            language=detected_language,
            language_probability=None,
            duration=duration,
        )
        logger.info(
            "[SenseVoice] 转写完成: %d 条字幕, language=%s, duration=%.1fs",
            len(subtitle_segments),
            transcription_info.language,
            transcription_info.duration,
        )
        return subtitle_segments, transcription_info

    def _ensure_sensevoice_models_available(self) -> None:
        """分别校验 SenseVoice 主模型与 VAD 模型的本地目录。"""
        self._ensure_local_model_dir(
            self._model_path,
            label="SenseVoice",
            required_any=("model.pt",),
            required_all=("config.yaml",),
        )
        self._ensure_local_model_dir(
            self._vad_path,
            label="FSMN-VAD",
            required_any=("model.pt",),
            required_all=("config.yaml",),
        )

    def _resolve_device(self) -> str:
        """将 CLI device 选项映射为 FunASR 所需的 device 字符串。"""
        if self._device == "cpu":
            return "cpu"
        if self._device == "cuda":
            return "cuda:0"

        # auto: 检测 PyTorch CUDA 可用性
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _map_language(language: Optional[str]) -> str:
        """
        将 ISO 语言代码映射为 SenseVoice ``generate`` 的 language 参数。

        SenseVoice 支持: auto, zh, en, yue, ja, ko, nospeech。
        """
        if language is None or language == "auto":
            return "auto"
        normalized = language.strip().lower()
        return _ISO_TO_SENSEVOICE_LANG.get(normalized, normalized)

    @staticmethod
    def _audio_duration_ms(audio_path: Path) -> int:
        """从 WAV 文件读取时长（毫秒）。"""
        import wave

        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return 0
            return int(frames / rate * 1000)

    @classmethod
    def _resolve_sentence_info(
        cls,
        result_item: dict,
        *,
        audio_path: Path,
        postprocess,
    ) -> list[dict]:
        """
        解析 FunASR 返回的分段信息。

        FunASR 在未配置 spk_model 时通常不填充 ``sentence_info``，
        需从 ``timestamp`` 或语言标签分段自行构建。
        """
        sentence_info = result_item.get("sentence_info") or []
        if sentence_info:
            return sentence_info

        timestamps = result_item.get("timestamp")
        text = result_item.get("text") or ""
        words = result_item.get("words")
        if isinstance(timestamps, list) and timestamps and text:
            built = cls._sentence_info_from_word_timestamps(
                text,
                timestamps,
                words if isinstance(words, list) else None,
            )
            if built:
                return built

        if text.strip():
            total_ms = cls._total_duration_ms(result_item, audio_path, timestamps)
            built = cls._sentence_info_from_lang_tags(
                text,
                total_ms,
                postprocess=postprocess,
            )
            if built:
                return built

        return []

    @staticmethod
    def _total_duration_ms(
        result_item: dict,
        audio_path: Path,
        timestamps: object,
    ) -> int:
        """估算音频总时长（毫秒），供无词级时间戳时的比例分配。"""
        if isinstance(timestamps, list) and timestamps:
            last = timestamps[-1]
            if isinstance(last, (list, tuple)) and len(last) >= 2:
                return int(last[1])

        for key in ("duration", "audio_duration", "speech_length"):
            value = result_item.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value if value > 10_000 else value * 1000)

        if audio_path.is_file():
            return SenseVoiceTranscriptionService._audio_duration_ms(audio_path)
        return 0

    @staticmethod
    def _sentence_info_from_word_timestamps(
        text: str,
        timestamps: list,
        words: list | None,
    ) -> list[dict]:
        """将 ``output_timestamp=True`` 合并后的词级时间戳切成字幕句。"""
        token_texts = (
            words if words and len(words) == len(timestamps) else [""] * len(timestamps)
        )
        segments: list[dict] = []
        buf_words: list[str] = []
        seg_start: int | None = None
        seg_end: int | None = None

        for idx, ts in enumerate(timestamps):
            if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                continue
            start_ms, end_ms = int(ts[0]), int(ts[1])
            word = token_texts[idx] if idx < len(token_texts) else ""
            word = word.replace("▁", "").strip()

            if seg_start is None:
                seg_start = start_ms
            seg_end = end_ms
            if word:
                buf_words.append(word)

            buf_text = "".join(buf_words)
            if not buf_text:
                continue

            if ends_sentence(word):
                segments.append(
                    {
                        "start": seg_start,
                        "end": seg_end,
                        "text": buf_text,
                    }
                )
                buf_words = []
                seg_start = None
                seg_end = None
                continue

        if buf_words and seg_start is not None and seg_end is not None:
            segments.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "text": "".join(buf_words),
                }
            )

        if not segments and text.strip():
            last_end = int(timestamps[-1][1]) if timestamps else 0
            segments.append({"start": 0, "end": last_end, "text": text})

        return segments

    @staticmethod
    def _sentence_info_from_lang_tags(
        text: str,
        total_duration_ms: int,
        *,
        postprocess,
        max_chars: int = 14,
    ) -> list[dict]:
        """按 SenseVoice 语言标签切分，并按字符数比例分配时间。"""
        parts = _SENSEVOICE_LANG_SPLIT.split(text)
        chunks: list[str] = []
        index = 1
        while index < len(parts) - 1:
            lang = parts[index]
            content = parts[index + 1]
            chunks.append(f"<|{lang}|>{content}")
            index += 2
        if not chunks:
            chunks = [text]

        expanded_chunks: list[str] = []
        for chunk in chunks:
            clean = postprocess(chunk).strip()
            if not clean:
                continue
            if len(clean) <= max_chars:
                expanded_chunks.append(chunk)
            else:
                sub_texts = split_text_chunks(clean, max_chars)
                for sub in sub_texts:
                    expanded_chunks.append(sub)

        clean_lengths = [len(postprocess(chunk).strip()) for chunk in expanded_chunks]
        total_chars = sum(clean_lengths) or 1
        if total_duration_ms <= 0:
            total_duration_ms = max(total_chars * 200, 1000)

        segments: list[dict] = []
        cursor = 0
        for chunk, clean_len in zip(expanded_chunks, clean_lengths):
            if clean_len <= 0:
                continue
            seg_ms = max(int(total_duration_ms * clean_len / total_chars), 500)
            start = cursor
            end = min(cursor + seg_ms, total_duration_ms)
            segments.append({"start": start, "end": end, "text": chunk})
            cursor = end

        if segments:
            segments[-1]["end"] = total_duration_ms
        return segments

    @staticmethod
    def _to_subtitle_segments(
        sentence_info: list[dict],
        *,
        postprocess,
    ) -> list[SubtitleSegment]:
        """
        将 FunASR ``sentence_info`` 转为 SubtitleSegment 列表。

        ``start`` / ``end`` 单位为毫秒，需除以 1000 转为秒。
        """
        segments: list[SubtitleSegment] = []
        for index, seg in enumerate(sentence_info, start=1):
            raw_text = seg.get("sentence") or seg.get("text") or ""
            clean_text = postprocess(raw_text).strip()
            if not clean_text:
                continue
            segments.append(
                SubtitleSegment(
                    index=index,
                    start=float(seg["start"]) / 1000.0,
                    end=float(seg["end"]) / 1000.0,
                    text=clean_text,
                )
            )
        # 过滤空行后重新编号，保证 index 连续
        return [
            SubtitleSegment(index=i, start=s.start, end=s.end, text=s.text)
            for i, s in enumerate(segments, start=1)
        ]

    @staticmethod
    def _detect_language(
        sentence_info: list[dict],
        *,
        fallback: Optional[str],
    ) -> Optional[str]:
        """从 SenseVoice 富文本标签或用户指定语言推断检测结果。"""
        for seg in sentence_info:
            raw_text = seg.get("sentence") or seg.get("text") or ""
            match = _SENSEVOICE_LANG_TAG.search(raw_text)
            if match:
                return match.group(1)
        return fallback

    @staticmethod
    def _estimate_duration(
        result_item: dict,
        segments: list[SubtitleSegment],
    ) -> float:
        """优先使用 FunASR 返回的时长，否则取最后一条字幕的结束时间。"""
        for key in ("duration", "audio_duration", "speech_length"):
            value = result_item.get(key)
            if isinstance(value, (int, float)) and value > 0:
                # FunASR 部分字段单位为毫秒
                if value > 10_000:
                    return float(value) / 1000.0
                return float(value)
        if segments:
            return segments[-1].end
        return 0.0


TranscriptionService = SenseVoiceTranscriptionService


def create_transcription_service(
    config: PipelineConfig,
) -> TranscriptionBackendProtocol:
    """根据 PipelineConfig 创建 SenseVoice 转写服务实例。"""
    return SenseVoiceTranscriptionService(
        model_path=config.sensevoice_model_path,
        vad_path=config.sensevoice_vad_path,
        device=config.device,
    )
