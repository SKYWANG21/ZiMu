"""语音转写封装：支持 faster-whisper、SenseVoice（FunASR）与远程 HTTP API。"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import uuid
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment as WhisperSegment

from src.models import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REMOTE_API_URL,
    DEFAULT_SENSEVOICE_MODEL_PATH,
    DEFAULT_SENSEVOICE_VAD_PATH,
    PipelineConfig,
    SubtitleSegment,
)
from src.segment import ends_sentence, split_text_chunks

logger = logging.getLogger(__name__)

DeviceChoice = Literal["auto", "cuda", "cpu"]

# SenseVoice 输出文本中的语言标签，如 <|zh|>、|<|en|>
_SENSEVOICE_LANG_TAG = re.compile(r"<\|([a-z]{2,10})\|>")
_SENSEVOICE_LANG_SPLIT = re.compile(r"<\|(zh|en|yue|ja|ko|nospeech)\|>")

# Whisper ISO 代码 -> SenseVoice language 参数（FunASR 文档约定）
_WHISPER_TO_SENSEVOICE_LANG: dict[str, str] = {
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
        校验本地模型目录存在且包含 FunASR / faster-whisper 所需的关键文件。

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


class WhisperTranscriptionService(BaseTranscriptionService):
    """faster-whisper 转写后端：懒加载 WhisperModel，支持 VAD 与 beam search。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: DeviceChoice = "auto",
    ) -> None:
        self._model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._device = device
        self._model: Optional[WhisperModel] = None

    @property
    def model(self) -> WhisperModel:
        """懒加载 Whisper 模型，避免未使用时占用内存。"""
        if self._model is None:
            self._ensure_whisper_model_available()
            resolved_device, compute_type = self._resolve_device_and_compute_type()
            logger.info(
                "加载 Whisper 模型: path=%s device=%s compute_type=%s",
                self._model_path,
                resolved_device,
                compute_type,
            )
            self._model = WhisperModel(
                str(self._model_path),
                device=resolved_device,
                compute_type=compute_type,
                local_files_only=True,
            )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
    ) -> tuple[list[SubtitleSegment], TranscriptionInfo]:
        """
        使用 faster-whisper 转写音频。

        Args:
            audio_path: 16 kHz 单声道 WAV 路径。
            language: ISO 语言代码（如 zh、en）；None 表示自动检测。
        """
        logger.info("[Whisper] 开始转写: %s", audio_path.name)
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
        )
        whisper_segments = list(segments_iter)
        subtitle_segments = self._to_subtitle_segments(whisper_segments)
        transcription_info = TranscriptionInfo(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )
        logger.info(
            "[Whisper] 转写完成: %d 条字幕, language=%s (%.2f), duration=%.1fs",
            len(subtitle_segments),
            transcription_info.language,
            transcription_info.language_probability or 0.0,
            transcription_info.duration,
        )
        return subtitle_segments, transcription_info

    def _ensure_whisper_model_available(self) -> None:
        """确认 faster-whisper 本地模型目录包含 CTranslate2 所需文件。"""
        if not self._model_path.is_dir():
            raise FileNotFoundError(
                f"Whisper 本地模型目录不存在: {self._model_path}\n"
                "请先将 faster-whisper 模型下载到该目录。"
            )
        required_files = ("config.json", "model.bin")
        missing = [
            name for name in required_files if not (self._model_path / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Whisper 本地模型目录不完整: {self._model_path}\n"
                f"缺少文件: {', '.join(missing)}"
            )

    def _resolve_device_and_compute_type(self) -> tuple[str, str]:
        """根据 CLI 配置解析 ctranslate2 的 device 与 compute_type。"""
        if self._device == "cpu":
            return "cpu", "int8"
        if self._device == "cuda":
            return "cuda", "float16"

        # auto: 检测 CUDA 可用性，不可用时回退 CPU
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
        return "cpu", "int8"

    @staticmethod
    def _to_subtitle_segments(
        whisper_segments: list[WhisperSegment],
    ) -> list[SubtitleSegment]:
        """将 faster-whisper Segment 转换为统一的 SubtitleSegment 领域模型。"""
        result: list[SubtitleSegment] = []
        index = 1
        for segment in whisper_segments:
            text = segment.text.strip()
            if not text:
                continue

            words = getattr(segment, "words", None)
            if words:
                for piece in WhisperTranscriptionService._segments_from_words(
                    words,
                    fallback_text=text,
                    segment_start=segment.start,
                    segment_end=segment.end,
                ):
                    result.append(
                        SubtitleSegment(
                            index=index,
                            start=piece.start,
                            end=piece.end,
                            text=piece.text,
                        )
                    )
                    index += 1
            else:
                result.append(
                    SubtitleSegment(
                        index=index,
                        start=segment.start,
                        end=segment.end,
                        text=text,
                    )
                )
                index += 1
        return result

    @staticmethod
    def _segments_from_words(
        words,
        *,
        fallback_text: str,
        segment_start: float,
        segment_end: float,
    ) -> list[SubtitleSegment]:
        """按词级时间戳将 Whisper 分段拆成更细粒度的中间片段。"""
        pieces: list[SubtitleSegment] = []
        buf_words: list[str] = []
        piece_start: float | None = None
        piece_end: float | None = None

        for word in words:
            word_text = getattr(word, "word", "") or ""
            word_text = word_text.strip()
            word_start = float(getattr(word, "start", segment_start))
            word_end = float(getattr(word, "end", word_start))

            if piece_start is None:
                piece_start = word_start
            piece_end = word_end
            if word_text:
                buf_words.append(word_text)

            buf_text = "".join(buf_words).strip()
            if not buf_text:
                continue

            if ends_sentence(word_text):
                pieces.append(
                    SubtitleSegment(
                        index=0,
                        start=piece_start,
                        end=piece_end,
                        text=buf_text,
                    )
                )
                buf_words = []
                piece_start = None
                piece_end = None

        if buf_words and piece_start is not None and piece_end is not None:
            pieces.append(
                SubtitleSegment(
                    index=0,
                    start=piece_start,
                    end=piece_end,
                    text="".join(buf_words).strip(),
                )
            )

        if pieces:
            return pieces

        return [
            SubtitleSegment(
                index=0,
                start=segment_start,
                end=segment_end,
                text=fallback_text,
            )
        ]


class SenseVoiceTranscriptionService(BaseTranscriptionService):
    """
    SenseVoice（FunASR）转写后端。

    依赖本地 SenseVoiceSmall 与 FSMN-VAD 模型目录，通过 VAD 分段获取
    ``sentence_info`` 中的毫秒级时间戳，再经 ``rich_transcription_postprocess``
    去除 <|zh|> 等富文本标签后写入 SRT。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        vad_path: str | Path | None = None,
        device: DeviceChoice = "auto",
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
            audio_path: 16 kHz 单声道 WAV 路径。
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
        将 Whisper 风格 ISO 代码映射为 SenseVoice ``generate`` 的 language 参数。

        SenseVoice 支持: auto, zh, en, yue, ja, ko, nospeech。
        """
        if language is None or language == "auto":
            return "auto"
        normalized = language.strip().lower()
        return _WHISPER_TO_SENSEVOICE_LANG.get(normalized, normalized)

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
        # 过滤空行后重新编号，保证 SRT index 连续
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


class RemoteTranscriptionService:
    """
    远程 HTTP 转写后端。

    向 OpenAI 兼容的 ``/v1/audio/transcriptions`` 接口发送
    ``multipart/form-data`` 请求，字段为 ``file``（音频）与可选 ``language``。
    额外附带 ``response_format=verbose_json`` 以获取带时间戳的分段结果。
    """

    def __init__(
        self,
        api_url: str = DEFAULT_REMOTE_API_URL,
        *,
        timeout_sec: float = 600.0,
    ) -> None:
        self._api_url = api_url
        self._timeout_sec = timeout_sec

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
    ) -> tuple[list[SubtitleSegment], TranscriptionInfo]:
        """
        调用远程转写 API。

        Args:
            audio_path: 音频文件路径（通常为 16 kHz WAV）。
            language: 可选语言代码，如 zh、en。
        """
        if not audio_path.is_file():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        logger.info("[Remote] 开始转写: %s -> %s", audio_path.name, self._api_url)
        form_fields: dict[str, str] = {}
        if language:
            form_fields["language"] = language

        raw_body = self._post_transcription(audio_path, form_fields)
        payload = self._parse_response_body(raw_body)
        segments = self._to_subtitle_segments(payload, audio_path=audio_path)
        duration = self._resolve_duration(payload, audio_path, segments)
        detected_language = payload.get("language") if isinstance(payload, dict) else None
        if detected_language is None and language:
            detected_language = language

        info = TranscriptionInfo(
            language=detected_language,
            language_probability=None,
            duration=duration,
        )
        logger.info(
            "[Remote] 转写完成: %d 条字幕, language=%s, duration=%.1fs",
            len(segments),
            info.language,
            info.duration,
        )
        return segments, info

    def _post_transcription(
        self,
        audio_path: Path,
        form_fields: dict[str, str],
    ) -> bytes:
        """
        请求远程转写接口。

        优先附带 ``response_format=verbose_json`` 以获取分段时间戳；
        若服务端不支持则回退为仅 ``file`` / ``language`` 字段。
        """
        fields_with_verbose = {**form_fields, "response_format": "verbose_json"}
        try:
            return self._post_multipart(
                self._api_url,
                file_field="file",
                file_path=audio_path,
                fields=fields_with_verbose,
            )
        except RuntimeError as exc:
            if "verbose_json" not in str(exc):
                raise
            logger.warning(
                "[Remote] 服务端不支持 verbose_json，回退为纯文本响应"
            )
            return self._post_multipart(
                self._api_url,
                file_field="file",
                file_path=audio_path,
                fields=form_fields,
            )

    def _post_multipart(
        self,
        url: str,
        *,
        file_field: str,
        file_path: Path,
        fields: dict[str, str],
    ) -> bytes:
        """以 multipart/form-data 向远程接口 POST 音频与表单字段。"""
        boundary = uuid.uuid4().hex
        body_parts: list[bytes] = []

        for name, value in fields.items():
            body_parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode()
            )

        file_data = file_path.read_bytes()
        filename = file_path.name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body_parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
            + file_data
            + b"\r\n"
        )
        body_parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(body_parts)

        request = Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urlopen(request, timeout=self._timeout_sec) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"远程转写 API 返回 HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"远程转写 API 请求失败: {exc.reason}") from exc

    @staticmethod
    def _parse_response_body(raw_body: bytes) -> dict[str, Any] | str:
        """解析 API 响应：优先 JSON，否则按纯文本处理。"""
        text = raw_body.decode("utf-8", errors="replace").strip()
        if not text:
            raise ValueError("远程转写 API 返回空响应")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(parsed, dict):
            return parsed
        return text

    def _to_subtitle_segments(
        self,
        payload: dict[str, Any] | str,
        *,
        audio_path: Path,
    ) -> list[SubtitleSegment]:
        """将 API 响应转为 SubtitleSegment 列表。"""
        if isinstance(payload, dict):
            raw_segments = payload.get("segments")
            if isinstance(raw_segments, list) and raw_segments:
                segments: list[SubtitleSegment] = []
                index = 1
                for item in raw_segments:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue
                    segments.append(
                        SubtitleSegment(
                            index=index,
                            start=float(item.get("start", 0.0)),
                            end=float(item.get("end", 0.0)),
                            text=text,
                        )
                    )
                    index += 1
                if segments:
                    return segments

            full_text = str(payload.get("text", "")).strip()
            if full_text:
                duration = self._read_wav_duration(audio_path)
                return [
                    SubtitleSegment(
                        index=1,
                        start=0.0,
                        end=duration,
                        text=full_text,
                    )
                ]

        if isinstance(payload, str) and payload.strip():
            duration = self._read_wav_duration(audio_path)
            return [
                SubtitleSegment(
                    index=1,
                    start=0.0,
                    end=duration,
                    text=payload.strip(),
                )
            ]

        raise ValueError("远程转写 API 未返回可用文本或分段结果")

    @staticmethod
    def _resolve_duration(
        payload: dict[str, Any] | str,
        audio_path: Path,
        segments: list[SubtitleSegment],
    ) -> float:
        """从响应、字幕片段或音频文件推断总时长（秒）。"""
        if isinstance(payload, dict):
            duration = payload.get("duration")
            if isinstance(duration, (int, float)) and duration > 0:
                return float(duration)
        if segments:
            return segments[-1].end
        return RemoteTranscriptionService._read_wav_duration(audio_path)

    @staticmethod
    def _read_wav_duration(audio_path: Path) -> float:
        """读取 WAV 文件时长；非 WAV 或解析失败时返回 0。"""
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                if sample_rate > 0:
                    return frame_count / float(sample_rate)
        except wave.Error:
            pass
        return 0.0


# 向后兼容：旧代码中的 TranscriptionService 即 Whisper 实现
TranscriptionService = WhisperTranscriptionService


def create_transcription_service(
    config: PipelineConfig,
) -> TranscriptionBackendProtocol:
    """
    根据 PipelineConfig.backend 创建对应的转写服务实例。

    Args:
        config: 流水线配置，含后端类型、模型路径与 device。

    Returns:
        实现了 transcribe 方法的转写服务。
    """
    if config.backend == "sensevoice":
        return SenseVoiceTranscriptionService(
            model_path=config.sensevoice_model_path,
            vad_path=config.sensevoice_vad_path,
            device=config.device,
        )
    if config.backend == "remote":
        return RemoteTranscriptionService(api_url=config.remote_api_url)
    return WhisperTranscriptionService(
        model_path=config.model_path,
        device=config.device,
    )
