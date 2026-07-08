"""字幕断句：将长片段按字数、时长与标点切分为更短的字幕条。"""

from __future__ import annotations

import re

from src.models import SubtitleSegment

_STRONG_BREAK_RE = re.compile(r"[。！？.!?…]")
_WEAK_BREAK_RE = re.compile(r"[，、；,:]")
_SENTENCE_END_RE = re.compile(r"[。！？.!?…]+$")
_MIN_SEGMENT_DURATION_SEC = 0.3


def split_text_chunks(text: str, max_chars: int = 14) -> list[str]:
    """
    将文本切分为不超过 max_chars 的片段，优先在标点处断开。

    断点优先级：句末标点 > 逗号等弱标点 > 空格 > 硬切。
    """
    text = text.strip()
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        window = remaining[:max_chars]
        break_at = _find_break_index(window)
        if break_at <= 0:
            break_at = max_chars

        chunk = remaining[:break_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[break_at:]

    return chunks


def _find_break_index(window: str) -> int:
    """在 window 内从后向前找最佳断点位置（不含该位置字符）。"""
    for pattern in (_STRONG_BREAK_RE, _WEAK_BREAK_RE):
        for match in reversed(list(pattern.finditer(window))):
            pos = match.end()
            if pos > 0:
                return pos

    space = window.rfind(" ")
    if space > 0:
        return space + 1

    return len(window)


def split_subtitle_segments(
    segments: list[SubtitleSegment],
    *,
    max_chars: int = 14,
    max_duration_sec: float = 3.5,
) -> list[SubtitleSegment]:
    """
    将过长字幕片段切分为更短的 SubtitleSegment 列表。

    对每条 segment，若字数或时长超限则按 split_text_chunks 切分文本，
    并按字符数比例线性分配时间轴。
    """
    if not segments:
        return []

    result: list[SubtitleSegment] = []
    for segment in segments:
        result.extend(
            _split_one_segment(
                segment,
                max_chars=max_chars,
                max_duration_sec=max_duration_sec,
            )
        )

    return [
        SubtitleSegment(index=i, start=s.start, end=s.end, text=s.text)
        for i, s in enumerate(result, start=1)
    ]


def _split_one_segment(
    segment: SubtitleSegment,
    *,
    max_chars: int,
    max_duration_sec: float,
) -> list[SubtitleSegment]:
    text = segment.text.strip()
    if not text:
        return []

    duration = segment.end - segment.start
    if len(text) <= max_chars and duration <= max_duration_sec:
        return [
            SubtitleSegment(
                index=segment.index,
                start=segment.start,
                end=segment.end,
                text=text,
            )
        ]

    chunks = split_text_chunks(text, max_chars)
    if len(chunks) <= 1 and duration <= max_duration_sec:
        return [
            SubtitleSegment(
                index=segment.index,
                start=segment.start,
                end=segment.end,
                text=text,
            )
        ]

    if len(chunks) <= 1:
        chunks = split_text_chunks(text, max(1, max_chars // 2))
        if len(chunks) <= 1:
            chunks = _split_by_duration(text, duration, max_duration_sec)

    return _assign_times_proportional(
        chunks,
        start=segment.start,
        end=segment.end,
        max_duration_sec=max_duration_sec,
    )


def _split_by_duration(
    text: str,
    duration: float,
    max_duration_sec: float,
) -> list[str]:
    """按时长估算需要的切分份数，再按字数均分文本。"""
    if duration <= max_duration_sec or max_duration_sec <= 0:
        return [text]
    parts = max(2, int(duration / max_duration_sec) + 1)
    chunk_size = max(1, (len(text) + parts - 1) // parts)
    return split_text_chunks(text, chunk_size)


def _assign_times_proportional(
    chunks: list[str],
    *,
    start: float,
    end: float,
    max_duration_sec: float,
) -> list[SubtitleSegment]:
    """按字符数比例分配时间，并保证末条对齐原 end。"""
    if not chunks:
        return []

    total_chars = sum(len(c) for c in chunks) or 1
    duration = end - start
    cursor = start
    result: list[SubtitleSegment] = []

    for index, chunk in enumerate(chunks):
        if index == len(chunks) - 1:
            chunk_end = end
        else:
            ratio = len(chunk) / total_chars
            chunk_duration = max(duration * ratio, _MIN_SEGMENT_DURATION_SEC)
            chunk_duration = min(chunk_duration, max_duration_sec)
            chunk_end = min(cursor + chunk_duration, end)

        if chunk_end <= cursor:
            chunk_end = min(cursor + _MIN_SEGMENT_DURATION_SEC, end)

        result.append(
            SubtitleSegment(
                index=0,
                start=cursor,
                end=chunk_end,
                text=chunk,
            )
        )
        cursor = chunk_end

    if result:
        result[-1] = SubtitleSegment(
            index=0,
            start=result[-1].start,
            end=end,
            text=result[-1].text,
        )

    return result


def ends_sentence(word: str) -> bool:
    """判断词/token 是否以句末标点结尾。"""
    return bool(_SENTENCE_END_RE.search(word))
