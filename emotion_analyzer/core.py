from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


EMOTION_LABELS = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
ProgressCallback = Callable[[int, str], None]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_WHISPER_MODEL = PROJECT_ROOT / "models" / "faster-whisper-large-v3"
DEFAULT_EMOTION_MODEL = "iic/emotion2vec_plus_large"
EMOTION_MODEL_OPTIONS = (
    ("emotion2vec+ large（約 300M）", DEFAULT_EMOTION_MODEL),
    ("emotion2vec+ base（約 90M）", "iic/emotion2vec_plus_base"),
    ("emotion2vec+ seed（學術語料版）", "iic/emotion2vec_plus_seed"),
)
LOCAL_EMOTION_MODELS = {
    model_name: PROJECT_ROOT / "models" / model_name.rsplit("/", 1)[-1]
    for _, model_name in EMOTION_MODEL_OPTIONS
}
DEFAULT_NOISE_REDUCTION_PROFILE = "speech_conservative_v1"
NOISE_REDUCTION_FILTERS = {
    DEFAULT_NOISE_REDUCTION_PROFILE: "afftdn=nr=8:nf=-50:tn=1:gs=5",
}
CACHE_SCHEMA_VERSION = 1


class AnalysisError(RuntimeError):
    """A user-facing analysis error."""


def _prefer_local_model(model_name: str, local_path: Path, aliases: set[str]) -> str:
    """Use a bundled model for known default aliases, with the remote name as fallback."""
    if model_name in aliases and local_path.is_dir():
        return str(local_path)
    return model_name


def _prefer_local_emotion_model(model_name: str) -> str:
    """Resolve known emotion2vec+ variants from models/ before using ModelScope."""
    canonical_name = model_name if "/" in model_name else f"iic/{model_name}"
    local_path = LOCAL_EMOTION_MODELS.get(canonical_name)
    if local_path is not None and local_path.is_dir():
        return str(local_path)
    return model_name


@dataclass(frozen=True)
class AnalysisConfig:
    reference_audio: Path
    segment_count: int
    transcript_file: Path
    source_directory: Path
    match_threshold: float = 0.30
    output_excel: Path = Path("analysis_result.xlsx")
    model_name: str = "large-v3"
    emotion_model_name: str = DEFAULT_EMOTION_MODEL
    segment_padding_seconds: float = 1.0
    recursive: bool = False
    noise_reduction_enabled: bool = True
    noise_reduction_profile: str = DEFAULT_NOISE_REDUCTION_PROFILE


@dataclass
class SegmentResult:
    index: int
    start: float
    end: float
    confidence: float
    scores: np.ndarray
    emotion: str
    top_score: float
    error: str | None = None
    distance: float = math.nan


@dataclass
class SegmentedAudio:
    """A friendly-support-style cached segment ready for emotion inference."""

    index: int
    start: float
    end: float
    confidence: float
    audio: np.ndarray | None
    error: str | None = None


@dataclass(frozen=True)
class TextWindowMatch:
    """One globally ordered transcript match, including why a match failed."""

    start: float
    end: float
    score: float
    relaxed: bool = False
    error: str | None = None


@dataclass
class FileAnalysisResult:
    audio_name: str
    segments: list[SegmentResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class _TimedText:
    text: str
    start: float
    end: float


def normalize_text(text: str) -> str:
    """Normalize text for stable matching while retaining CJK characters."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char
        for char in normalized
        if not char.isspace() and (char.isalnum() or "\u3400" <= char <= "\u9fff")
    )


_STRUCTURED_TRANSCRIPT_ITEM = re.compile(
    r'''\(\s*(?:"(?P<straight>(?:\\.|[^"\\])*)["”]|“(?P<curly>.*?)[”"]|「(?P<cjk>.*?)」)\s*\)''',
    re.DOTALL,
)


def parse_transcript_text(text: str) -> list[str]:
    """Parse either one-segment-per-line text or {("segment"),(“segment”)} text."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        body = stripped[1:-1]
        segments: list[str] = []
        cursor = 0
        for match in _STRUCTURED_TRANSCRIPT_ITEM.finditer(body):
            separator = body[cursor : match.start()]
            expected_separator = r"\s*" if not segments else r"\s*,\s*"
            if re.fullmatch(expected_separator, separator) is None:
                raise AnalysisError(
                    "分段文字格式錯誤；請使用 {(\"第一段\"),(“第二段”)}，段落之間以逗號分隔。"
                )
            value = next(group for group in match.groups() if group is not None)
            if match.group("straight") is not None:
                value = value.replace(r"\\", "\\").replace(r'\"', '"')
            segments.append(value.strip())
            cursor = match.end()

        if not segments or re.fullmatch(r"\s*,?\s*", body[cursor:]) is None:
            raise AnalysisError(
                "分段文字格式錯誤；請使用 {(\"第一段\"),(“第二段”)}，段落之間以逗號分隔。"
            )
    else:
        segments = [line.strip() for line in text.splitlines() if line.strip()]

    if not segments:
        raise AnalysisError("分段文字檔沒有可用的段落。")
    if any(not normalize_text(segment) for segment in segments):
        raise AnalysisError("文字檔中存在只包含標點或空白的段落。")
    return segments


def read_transcript(path: Path, expected_count: int | None = None) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AnalysisError(f"文字檔不是 UTF-8 編碼：{path.name}") from exc
    lines = parse_transcript_text(text)
    if expected_count is not None and len(lines) != expected_count:
        raise AnalysisError(
            f"分段數量為 {expected_count}，但文字檔解析出 {len(lines)} 個段落。"
        )
    return lines


def _natural_key(path: Path) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def enumerate_audio_files(directory: Path, recursive: bool = False) -> list[Path]:
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.casefold() in SUPPORTED_AUDIO_EXTENSIONS]
    return sorted(files, key=_natural_key)


def clear_chopped_directories(source_directories: Sequence[Path]) -> list[Path]:
    """Remove only the ``chopped`` cache directly inside each selected directory."""
    removed: list[Path] = []
    seen: set[str] = set()
    for source_directory in source_directories:
        source = Path(source_directory).resolve()
        key = str(source).casefold()
        if key in seen:
            continue
        seen.add(key)
        if not source.is_dir():
            raise AnalysisError(f"找不到待清除的資料夾：{source}")
        cache = (source / "chopped").resolve()
        if cache.parent != source or cache.name.casefold() != "chopped":
            raise AnalysisError(f"拒絕清除非 chopped 路徑：{cache}")
        if cache.is_dir():
            shutil.rmtree(cache)
            removed.append(cache)
    return removed


def _find_ffmpeg() -> str:
    bundled = Path(__file__).resolve().parent / "resources" / "ffmpeg.exe"
    if bundled.is_file():
        return str(bundled)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise AnalysisError("找不到 FFmpeg。請將 ffmpeg.exe 放入 resources，或加入 PATH。")


def _decode_audio(
    path: Path,
    ffmpeg_path: str | None = None,
    *,
    audio_filter: str | None = None,
) -> np.ndarray:
    command = [
        ffmpeg_path or _find_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
    ]
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.extend(
        [
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ]
    )
    try:
        completed = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise AnalysisError("找不到 FFmpeg。") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise AnalysisError(f"無法解碼音檔 {path.name}：{message}") from exc
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise AnalysisError(f"音檔沒有可分析的音訊：{path.name}")
    return audio


def decode_audio(path: Path, ffmpeg_path: str | None = None) -> np.ndarray:
    """Decode audio without enhancement, primarily for processed cache files."""
    return _decode_audio(path, ffmpeg_path)


def preprocess_audio(
    path: Path,
    ffmpeg_path: str | None = None,
    *,
    noise_reduction_enabled: bool = True,
    noise_reduction_profile: str = DEFAULT_NOISE_REDUCTION_PROFILE,
) -> np.ndarray:
    """Decode a source file and apply configured enhancement exactly once."""
    audio_filter = None
    if noise_reduction_enabled:
        try:
            audio_filter = NOISE_REDUCTION_FILTERS[noise_reduction_profile]
        except KeyError as exc:
            raise AnalysisError(f"不支援的去雜音模式：{noise_reduction_profile}") from exc
    return _decode_audio(path, ffmpeg_path, audio_filter=audio_filter)


def _write_temp_wav(audio: np.ndarray) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = handle.name
    handle.close()
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm.tobytes())
    return path


def _write_wav_file(path: Path, audio: np.ndarray) -> None:
    """Write a normalized 16 kHz mono PCM cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm.tobytes())


def _expand_timed_text(text: str, start: float, end: float) -> list[_TimedText]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    duration = max(0.0, end - start)
    result: list[_TimedText] = []
    for index, char in enumerate(normalized):
        char_start = start + duration * index / len(normalized)
        char_end = start + duration * (index + 1) / len(normalized)
        result.append(_TimedText(char, char_start, char_end))
    return result


def _whisper_timed_chars(result: dict) -> list[_TimedText]:
    timed: list[_TimedText] = []
    for segment in result.get("segments", []):
        words = segment.get("words") or []
        if words:
            for word in words:
                timed.extend(
                    _expand_timed_text(
                        str(word.get("word", "")),
                        float(word.get("start", segment.get("start", 0.0))),
                        float(word.get("end", segment.get("end", 0.0))),
                    )
                )
        else:
            timed.extend(
                _expand_timed_text(
                    str(segment.get("text", "")),
                    float(segment.get("start", 0.0)),
                    float(segment.get("end", 0.0)),
                )
            )
    return timed


def _line_ranges(lines: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    normalized_lines = [normalize_text(line) for line in lines]
    text_parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in normalized_lines:
        text_parts.append(line)
        ranges.append((offset, offset + len(line)))
        offset += len(line)
    return "".join(text_parts), ranges


def _alignment_mapping(expected: str, observed: str) -> tuple[dict[int, int], list[tuple[str, int, int, int, int]]]:
    matcher = difflib.SequenceMatcher(None, expected, observed, autojunk=False)
    mapping: dict[int, int] = {}
    opcodes = matcher.get_opcodes()
    for tag, expected_start, expected_end, observed_start, observed_end in opcodes:
        expected_len = expected_end - expected_start
        observed_len = observed_end - observed_start
        if tag == "equal":
            for offset in range(expected_len):
                mapping[expected_start + offset] = observed_start + offset
        elif tag == "replace" and expected_len and observed_len:
            for offset in range(expected_len):
                mapped = observed_start + round(offset * (observed_len - 1) / max(1, expected_len - 1))
                mapping[expected_start + offset] = mapped
    return mapping, opcodes


def align_transcript(
    lines: Sequence[str],
    timed_chars: Sequence[_TimedText],
    threshold: float,
    audio_duration: float,
    padding_seconds: float = 0.15,
) -> list[tuple[float, float, float]]:
    expected, ranges = _line_ranges(lines)
    observed = "".join(item.text for item in timed_chars)
    if not observed:
        raise AnalysisError("Whisper 沒有產生可用的逐字時間戳。")
    mapping, _ = _alignment_mapping(expected, observed)
    aligned: list[tuple[float, float, float]] = []
    for line_start, line_end in ranges:
        line_length = line_end - line_start
        matched_indices = [mapping[i] for i in range(line_start, line_end) if i in mapping]
        confidence = len(matched_indices) / max(1, line_length)
        if not matched_indices or confidence < threshold:
            aligned.append((math.nan, math.nan, confidence))
            continue
        first = timed_chars[min(matched_indices)]
        last = timed_chars[max(matched_indices)]
        start = max(0.0, first.start - padding_seconds)
        end = min(audio_duration, last.end + padding_seconds)
        if end <= start:
            aligned.append((math.nan, math.nan, confidence))
        else:
            aligned.append((start, end, confidence))

    # Ensure adjacent segments do not overlap after padding.
    for index in range(len(aligned) - 1):
        start, end, confidence = aligned[index]
        next_start, next_end, next_confidence = aligned[index + 1]
        if math.isfinite(end) and math.isfinite(next_start) and end > next_start:
            midpoint = (end + next_start) / 2.0
            aligned[index] = (start, midpoint, confidence)
            aligned[index + 1] = (midpoint, next_end, next_confidence)
    return aligned


@lru_cache(maxsize=1)
def _get_s2t_converter():
    try:
        from opencc import OpenCC

        return OpenCC("s2t")
    except ImportError:
        return None


def _friendly_normalize(text: str) -> str:
    converter = _get_s2t_converter()
    if converter is not None:
        text = converter.convert(text)
    return re.sub(r"[^\w]", "", text).replace("_", "").casefold()


def _friendly_ratio(clean_text: str, clean_target: str) -> float:
    if not clean_target:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.ratio(clean_text, clean_target)) / 100.0
    except ImportError:
        return difflib.SequenceMatcher(None, clean_text, clean_target).ratio()


def _friendly_similarity(text: str, target: str) -> float:
    """Match friendly support's OpenCC + rapidfuzz ratio calculation."""
    return _friendly_ratio(_friendly_normalize(text), _friendly_normalize(target))


def _friendly_clean_text(text: str) -> str:
    return re.sub(r"[^\w]", "", text).replace("_", "")


def _friendly_window_candidates(
    timed_segments: Sequence[dict[str, object]],
    target: str,
    normalized_segments: Sequence[str] | None = None,
) -> list[TextWindowMatch]:
    """Return the best matching end point for every possible word start."""
    candidates: list[TextWindowMatch] = []
    clean_target = _friendly_normalize(target)
    target_clean_len = len(clean_target)
    clean_segments = list(normalized_segments) if normalized_segments is not None else [
        _friendly_normalize(str(segment["text"])) for segment in timed_segments
    ]
    max_window = max(target_clean_len * 3, 60)
    for start_index, segment in enumerate(timed_segments):
        start = float(segment["start"])
        best_score = 0.0
        best_end: float | None = None
        combined_clean = ""
        for end_index in range(start_index, min(start_index + max_window, len(timed_segments))):
            combined_clean += clean_segments[end_index]
            score = _friendly_ratio(combined_clean, clean_target)
            if score > best_score:
                best_score = score
                best_end = float(timed_segments[end_index]["end"])
            if len(combined_clean) > target_clean_len * 2 + 10:
                break
        if best_end is not None:
            candidates.append(TextWindowMatch(start, best_end, best_score))
    return candidates


def align_friendly_windows(
    timed_segments: Sequence[dict[str, object]],
    paragraphs: Sequence[tuple[str, str]],
    threshold: float,
) -> list[TextWindowMatch]:
    """Match the whole script in reading order, then safely recover near-threshold gaps."""
    normalized_segments = [
        _friendly_normalize(str(segment["text"])) for segment in timed_segments
    ]
    candidate_groups: list[list[TextWindowMatch]] = []
    for sentence_start, sentence_end in paragraphs:
        target = sentence_start if sentence_start == sentence_end else sentence_start + sentence_end
        candidate_groups.append(
            _friendly_window_candidates(timed_segments, target, normalized_segments)
        )

    matches: list[TextWindowMatch | None]
    overlap_tolerance = 0.35

    # First pass: dynamic programming selects the path that matches the most
    # paragraphs in order, then uses total similarity and earlier paragraphs as ties.
    states: list[tuple[float, int, float, int, tuple[TextWindowMatch | None, ...]]] = [
        (0.0, 0, 0.0, 0, ())
    ]
    paragraph_count = len(paragraphs)
    for index, candidates in enumerate(candidate_groups):
        next_states = [
            (last_end, count, score, priority, path + (None,))
            for last_end, count, score, priority, path in states
        ]
        for candidate in candidates:
            if candidate.score < threshold:
                continue
            eligible = [
                state for state in states if candidate.start + overlap_tolerance >= state[0]
            ]
            if not eligible:
                continue
            previous = max(eligible, key=lambda state: (state[1], state[2], state[3], -state[0]))
            next_states.append(
                (
                    candidate.end,
                    previous[1] + 1,
                    previous[2] + candidate.score,
                    previous[3] + (1 << (paragraph_count - index)),
                    previous[4] + (candidate,),
                )
            )
        states = next_states
    best_state = max(states, key=lambda state: (state[1], state[2], state[3], -state[0]))
    matches = list(best_state[4])
    strict_indices = {index for index, match in enumerate(matches) if match is not None}

    # Second pass: a slightly weak match is accepted only beside a strict anchor and
    # inside the time interval established by the surrounding strong matches.
    relaxed_threshold = max(0.20, threshold - 0.05)
    for index, candidates in enumerate(candidate_groups):
        if matches[index] is not None:
            continue
        previous_index = next(
            (candidate_index for candidate_index in range(index - 1, -1, -1) if matches[candidate_index]),
            None,
        )
        next_index = next(
            (candidate_index for candidate_index in range(index + 1, len(matches)) if candidate_index in strict_indices),
            None,
        )
        has_strict_anchor = (
            previous_index in strict_indices if previous_index is not None else False
        ) or next_index is not None
        if not has_strict_anchor:
            continue
        left = matches[previous_index].end if previous_index is not None else 0.0
        right = matches[next_index].start if next_index is not None else math.inf
        eligible = [
            candidate
            for candidate in candidates
            if candidate.score >= relaxed_threshold
            and candidate.start + overlap_tolerance >= left
            and candidate.end <= right + overlap_tolerance
        ]
        if not eligible:
            continue
        chosen = min(eligible, key=lambda candidate: (-candidate.score, candidate.start))
        matches[index] = TextWindowMatch(
            chosen.start,
            chosen.end,
            chosen.score,
            relaxed=True,
        )

    results: list[TextWindowMatch] = []
    for index, (match, candidates) in enumerate(zip(matches, candidate_groups)):
        if match is not None:
            results.append(match)
            continue
        best_score = max((candidate.score for candidate in candidates), default=0.0)
        if best_score >= threshold:
            reason = "文字配對順序衝突"
        elif best_score >= relaxed_threshold:
            reason = "文字配對上下文不足"
        else:
            reason = "文字相似度不足"
        results.append(
            TextWindowMatch(
                math.nan,
                math.nan,
                best_score,
                error=f"{reason}（最高相似度 {best_score:.3f}，門檻 {threshold:.3f}）",
            )
        )
    return results


def _audio_rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -180.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-9))


def _covered_seconds(segments: Sequence[dict[str, object]]) -> float:
    ranges = sorted(
        (float(segment["start"]), float(segment["end"]))
        for segment in segments
        if float(segment["end"]) > float(segment["start"])
    )
    if not ranges:
        return 0.0
    covered = 0.0
    left, right = ranges[0]
    for start, end in ranges[1:]:
        if start <= right:
            right = max(right, end)
        else:
            covered += right - left
            left, right = start, end
    return covered + right - left


def _write_whisper_diagnostics(
    audio_path: Path,
    audio_duration: float,
    rms_dbfs: float,
    raw_segments: Sequence[dict[str, object]],
    sentence_segments: Sequence[dict[str, object]],
    timed_segments: Sequence[dict[str, object]],
) -> float:
    cache_dir = audio_path.parent / "chopped"
    cache_dir.mkdir(parents=True, exist_ok=True)
    coverage = _covered_seconds(sentence_segments) / max(audio_duration, 1e-9)

    raw_path = cache_dir / f"{audio_path.stem}_whisper_raw.txt"
    with raw_path.open("w", encoding="utf-8") as stream:
        stream.write(f"audio_duration_seconds={audio_duration:.3f}\n")
        stream.write(f"audio_rms_dbfs={rms_dbfs:.3f}\n")
        stream.write(f"raw_segment_count={len(raw_segments)}\n")
        stream.write(f"kept_segment_count={len(sentence_segments)}\n")
        stream.write(f"kept_time_coverage={coverage:.3f}\n\n")
        stream.write("=== raw segments and filter decisions ===\n")
        for segment in raw_segments:
            decision = "kept" if not segment.get("filter_reason") else f"rejected:{segment['filter_reason']}"
            stream.write(
                f"[{float(segment['start']):.2f}s - {float(segment['end']):.2f}s] "
                f"[{decision}] {segment['text']}\n"
            )

    transcript_path = cache_dir / f"{audio_path.stem}_whisper_transcript.txt"
    with transcript_path.open("w", encoding="utf-8") as stream:
        for segment in sentence_segments:
            stream.write(f"[{float(segment['start']):.2f}s - {float(segment['end']):.2f}s] {segment['text']}\n")
        stream.write("\n=== word timestamps ===\n")
        for segment in timed_segments:
            stream.write(f"[{float(segment['start']):.2f}s - {float(segment['end']):.2f}s] {segment['text']}\n")
        stream.write("\n=== diagnostics ===\n")
        stream.write(f"audio_rms_dbfs={rms_dbfs:.3f}\n")
        stream.write(f"kept_time_coverage={coverage:.3f}\n")
    return coverage


def _object_value(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class WhisperSegmenter:
    """Faster-Whisper segmenter matching friendly support's window search."""

    def __init__(self, model_name: str = "large-v3", device: str | None = None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AnalysisError("缺少 faster-whisper，請先安裝 requirements.txt。") from exc
        self._prepare_cuda_dll_paths()
        try:
            import torch

            detected_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            detected_device = "cpu"
        self.device = device or detected_device
        compute_type = "float16" if self.device.startswith("cuda") else "int8"
        model_source = _prefer_local_model(
            model_name,
            LOCAL_WHISPER_MODEL,
            {"large-v3", "Systran/faster-whisper-large-v3"},
        )
        try:
            self.model = WhisperModel(model_source, device=self.device, compute_type=compute_type)
        except Exception as exc:
            if self.device.startswith("cuda"):
                self.device = "cpu"
                self.model = WhisperModel(model_source, device="cpu", compute_type="int8")
            else:
                raise AnalysisError(f"Whisper 模型載入失敗：{exc}") from exc

    @staticmethod
    def _prepare_cuda_dll_paths() -> None:
        """Make pip-installed NVIDIA DLLs discoverable on Windows."""
        try:
            import site

            add_dll_directory = getattr(os, "add_dll_directory", None)
            for site_package in site.getsitepackages():
                for relative in (
                    Path("nvidia") / "cublas" / "bin",
                    Path("nvidia") / "cudnn" / "bin",
                ):
                    dll_path = Path(site_package) / relative
                    if not dll_path.is_dir():
                        continue
                    if add_dll_directory:
                        add_dll_directory(str(dll_path))
                    path_value = os.environ.get("PATH", "")
                    if str(dll_path) not in path_value.split(os.pathsep):
                        os.environ["PATH"] = str(dll_path) + os.pathsep + path_value
        except Exception:
            # CPU-only installations do not have these directories.
            return

    @staticmethod
    def _cache_paths(audio_path: Path, segment_count: int) -> list[Path]:
        cache_dir = audio_path.parent / "chopped"
        return [cache_dir / f"{audio_path.stem}_段落{index}.wav" for index in range(1, segment_count + 1)]

    @staticmethod
    def _cache_manifest_path(audio_path: Path) -> Path:
        return audio_path.parent / "chopped" / f"{audio_path.stem}_cache.json"

    @staticmethod
    def _cache_manifest(
        audio_path: Path,
        paragraphs: Sequence[tuple[str, str]],
        threshold: float,
        padding_seconds: float,
        noise_reduction_enabled: bool,
        noise_reduction_profile: str,
    ) -> dict[str, object]:
        stat = audio_path.stat()
        transcript_payload = json.dumps(
            list(paragraphs), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_path": str(audio_path.resolve()),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "segment_count": len(paragraphs),
            "transcript_sha256": hashlib.sha256(transcript_payload).hexdigest(),
            "match_threshold": threshold,
            "segment_padding_seconds": padding_seconds,
            "noise_reduction_enabled": noise_reduction_enabled,
            "noise_reduction_profile": noise_reduction_profile,
        }

    @staticmethod
    def _read_cache_manifest(path: Path) -> dict[str, object] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write_cache_manifest(path: Path, manifest: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def segment(
        self,
        audio_path: Path,
        audio: np.ndarray,
        paragraphs: Sequence[tuple[str, str]],
        threshold: float,
        padding_seconds: float,
        ffmpeg_path: str | None = None,
        noise_reduction_enabled: bool = True,
        noise_reduction_profile: str = DEFAULT_NOISE_REDUCTION_PROFILE,
    ) -> list[SegmentedAudio]:
        cache_paths = self._cache_paths(audio_path, len(paragraphs))
        manifest_path = self._cache_manifest_path(audio_path)
        expected_manifest = self._cache_manifest(
            audio_path,
            paragraphs,
            threshold,
            padding_seconds,
            noise_reduction_enabled,
            noise_reduction_profile,
        )
        if (
            all(path.is_file() for path in cache_paths)
            and self._read_cache_manifest(manifest_path) == expected_manifest
        ):
            try:
                cached = []
                for index, path in enumerate(cache_paths):
                    cached_audio = decode_audio(path, ffmpeg_path)
                    cached.append(
                        SegmentedAudio(index, 0.0, len(cached_audio) / 16000.0, 1.0, cached_audio)
                    )
            except AnalysisError:
                manifest_path.unlink(missing_ok=True)
            else:
                return cached
        else:
            manifest_path.unlink(missing_ok=True)

        segments_gen, _ = self.model.transcribe(
            audio,
            word_timestamps=True,
            language="zh",
            initial_prompt=" ".join(start for start, _ in paragraphs),
            beam_size=5,
            condition_on_previous_text=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=5,
            hallucination_silence_threshold=1.5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        converter = _get_s2t_converter()

        raw_segments: list[dict[str, object]] = []
        sentence_segments: list[dict[str, object]] = []
        timed_segments: list[dict[str, object]] = []
        previous_texts: list[str] = []
        for segment in segments_gen:
            raw_text = str(_object_value(segment, "text", "") or "")
            segment_text = converter.convert(raw_text) if converter else raw_text
            start = float(_object_value(segment, "start", 0.0) or 0.0)
            end = float(_object_value(segment, "end", start) or start)
            clean_text = _friendly_clean_text(segment_text)
            filter_reason: str | None = None
            if not clean_text:
                filter_reason = "empty_text"
            else:
                chinese_count = len(re.findall(r"[\u4e00-\u9fff]", clean_text))
                if chinese_count / len(clean_text) < 0.5:
                    filter_reason = "low_chinese_ratio"
            if filter_reason is None and any(
                _friendly_similarity(segment_text, previous) > 0.90
                for previous in previous_texts[-5:]
            ):
                filter_reason = "near_duplicate"
            raw_segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": segment_text,
                    "filter_reason": filter_reason,
                }
            )
            if filter_reason is None:
                previous_texts.append(segment_text)
                sentence_segments.append({"start": start, "end": end, "text": segment_text})
            words = _object_value(segment, "words", None) or []
            if words:
                for word in words:
                    word_text = str(_object_value(word, "word", "") or "")
                    word_text = converter.convert(word_text) if converter else word_text
                    word_clean = _friendly_clean_text(word_text)
                    if not word_clean:
                        continue
                    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", word_clean))
                    if chinese_count == 0 and len(word_clean) > 3:
                        continue
                    timed_segments.append(
                        {
                            "start": float(_object_value(word, "start", start) or start),
                            "end": float(_object_value(word, "end", end) or end),
                            "text": word_text,
                        }
                    )
            elif filter_reason is None:
                timed_segments.append({"start": start, "end": end, "text": segment_text})

        audio_duration = len(audio) / 16000.0
        rms_dbfs = _audio_rms_dbfs(audio)
        coverage = _write_whisper_diagnostics(
            audio_path,
            audio_duration,
            rms_dbfs,
            raw_segments,
            sentence_segments,
            timed_segments,
        )
        if not timed_segments:
            if rms_dbfs <= -45.0:
                detail = f"音量過低：{rms_dbfs:.1f} dBFS"
            elif raw_segments and all(segment.get("filter_reason") for segment in raw_segments):
                reasons = sorted({str(segment["filter_reason"]) for segment in raw_segments})
                detail = "原始片段全部遭過濾：" + ", ".join(reasons)
            elif not raw_segments:
                detail = "VAD／Whisper 沒有偵測到語音"
            else:
                detail = "沒有可用的逐詞時間戳"
            raise AnalysisError(f"Whisper 沒有產生可用的語音文字片段（{detail}）。")

        results: list[SegmentedAudio] = []
        matches = align_friendly_windows(timed_segments, paragraphs, threshold)
        for index, match in enumerate(matches):
            if match.error is not None:
                quality_notes: list[str] = []
                if rms_dbfs <= -45.0:
                    quality_notes.append(f"音量偏低 {rms_dbfs:.1f} dBFS")
                if coverage < 0.40:
                    quality_notes.append(f"Whisper 時間覆蓋率偏低 {coverage:.0%}")
                error = match.error
                if quality_notes:
                    error += "；" + "；".join(quality_notes)
                results.append(SegmentedAudio(index, math.nan, math.nan, match.score, None, error))
                continue
            start = max(0.0, match.start - padding_seconds)
            end = min(audio_duration, match.end + padding_seconds)
            start_index = int(start * 16000)
            end_index = min(len(audio), max(start_index + 1, int(end * 16000)))
            segment_audio = audio[start_index:end_index]
            _write_wav_file(cache_paths[index], segment_audio)
            results.append(SegmentedAudio(index, start, end, match.score, segment_audio))
        if all(result.audio is not None for result in results) and all(
            path.is_file() for path in cache_paths
        ):
            self._write_cache_manifest(manifest_path, expected_manifest)
        return results

    def close(self) -> None:
        """Release Faster-Whisper resources before loading emotion2vec+."""
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class Emotion2VecClassifier:
    def __init__(self, model_name: str = DEFAULT_EMOTION_MODEL, device: str | None = None):
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise AnalysisError("缺少 FunASR，請先安裝 requirements.txt。") from exc
        model_source = _prefer_local_emotion_model(model_name)
        kwargs = {"model": model_source}
        if model_source == model_name:
            kwargs["hub"] = "ms"
        else:
            kwargs["disable_update"] = True
            kwargs["disable_pbar"] = True
        if device:
            kwargs["device"] = device
        self.model = AutoModel(**kwargs)

    def classify(self, audio: np.ndarray) -> tuple[np.ndarray, str, float]:
        temp_path = _write_temp_wav(audio)
        try:
            result = self.model.generate(
                temp_path,
                granularity="utterance",
                extract_embedding=False,
            )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        if isinstance(result, tuple):
            result = result[0]
        if not result:
            raise AnalysisError("emotion2vec+ 沒有回傳結果。")
        item = result[0] if isinstance(result, list) else result
        labels = [self._canonical_label(label) for label in item.get("labels", [])]
        scores = [float(score) for score in item.get("scores", [])]
        score_by_label = dict(zip(labels, scores))
        if any(label not in score_by_label for label in EMOTION_LABELS):
            raise AnalysisError("emotion2vec+ 回傳的情緒標籤不完整。")
        vector = np.array([score_by_label[label] for label in EMOTION_LABELS], dtype=np.float64)
        top_index = int(np.argmax(vector))
        return vector, EMOTION_LABELS[top_index], float(vector[top_index])

    @staticmethod
    def _canonical_label(label: str) -> str:
        value = str(label).split("/")[-1].strip().casefold()
        return "unknown" if value in {"<unk>", "unk"} else value


def l2_distance(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    return float(np.linalg.norm(left - right))


def _safe_error_text(error: str | None) -> str:
    return error or ""


class BatchAnalyzer:
    def __init__(
        self,
        whisper_segmenter: WhisperSegmenter | None = None,
        emotion_classifier: Emotion2VecClassifier | None = None,
        ffmpeg_path: str | None = None,
    ):
        self.whisper_segmenter = whisper_segmenter
        self.emotion_classifier = emotion_classifier
        self.ffmpeg_path = ffmpeg_path

    def run(
        self,
        config: AnalysisConfig,
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[FileAnalysisResult]:
        return self.run_many([config], progress, cancel_event)[0]

    def run_many(
        self,
        configs: Sequence[AnalysisConfig],
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[list[FileAnalysisResult]]:
        """Analyze one experiment spread across folders while loading each model once."""
        if not configs:
            raise AnalysisError("至少需要一個待分析資料夾。")
        for config in configs:
            self._validate_config(config)
        self._validate_shared_settings(configs)

        primary = configs[0]
        transcript_lines = read_transcript(primary.transcript_file, primary.segment_count)
        reference = primary.reference_audio.resolve()
        candidate_files: list[list[Path]] = []
        for config in configs:
            files = enumerate_audio_files(config.source_directory, config.recursive)
            candidate_files.append([path for path in files if path.resolve() != reference])

        created_whisper = self.whisper_segmenter is None
        if created_whisper:
            self.whisper_segmenter = WhisperSegmenter(primary.model_name)
        assert self.whisper_segmenter is not None
        release_whisper = created_whisper or isinstance(self.whisper_segmenter, WhisperSegmenter)
        paragraphs = [(line, line) for line in transcript_lines]
        prepared_reference: tuple[Path, np.ndarray, list[SegmentedAudio]] | None = None
        prepared_groups: list[list[tuple[Path, np.ndarray, list[SegmentedAudio]] | FileAnalysisResult]] = [
            [] for _ in configs
        ]
        total_audio = 1 + sum(len(files) for files in candidate_files)
        processed_audio = 0

        # Friendly Support preprocesses every file first and releases Whisper before
        # loading emotion2vec+. This avoids keeping both large models in GPU memory.
        try:
            self._check_cancel(cancel_event)
            self._report(progress, 0, f"預處理並切段標準音檔：{reference.name}")
            try:
                audio = preprocess_audio(
                    reference,
                    self.ffmpeg_path,
                    noise_reduction_enabled=primary.noise_reduction_enabled,
                    noise_reduction_profile=primary.noise_reduction_profile,
                )
                boundaries = self.whisper_segmenter.segment(
                    reference,
                    audio,
                    paragraphs,
                    primary.match_threshold,
                    primary.segment_padding_seconds,
                    self.ffmpeg_path,
                    primary.noise_reduction_enabled,
                    primary.noise_reduction_profile,
                )
            except AnalysisError as exc:
                raise AnalysisError(f"標準音檔無法分段：{exc}") from exc
            prepared_reference = (reference, audio, boundaries)
            processed_audio += 1

            for group_index, (config, files) in enumerate(zip(configs, candidate_files)):
                folder_label = config.source_directory.name or str(config.source_directory)
                for audio_path in files:
                    self._check_cancel(cancel_event)
                    percent = int(processed_audio * 45 / max(1, total_audio))
                    self._report(progress, percent, f"預處理並切段 [{folder_label}]：{audio_path.name}")
                    try:
                        audio = preprocess_audio(
                            audio_path,
                            self.ffmpeg_path,
                            noise_reduction_enabled=config.noise_reduction_enabled,
                            noise_reduction_profile=config.noise_reduction_profile,
                        )
                        boundaries = self.whisper_segmenter.segment(
                            audio_path,
                            audio,
                            paragraphs,
                            config.match_threshold,
                            config.segment_padding_seconds,
                            self.ffmpeg_path,
                            config.noise_reduction_enabled,
                            config.noise_reduction_profile,
                        )
                    except AnalysisError as exc:
                        prepared_groups[group_index].append(
                            FileAnalysisResult(audio_path.name, error=str(exc))
                        )
                    else:
                        prepared_groups[group_index].append((audio_path, audio, boundaries))
                    processed_audio += 1
        finally:
            if release_whisper:
                close = getattr(self.whisper_segmenter, "close", None)
                if close:
                    close()
                self.whisper_segmenter = None

        if self.emotion_classifier is None:
            self.emotion_classifier = Emotion2VecClassifier(primary.emotion_model_name)

        assert prepared_reference is not None
        reference_path, reference_audio, reference_boundaries = prepared_reference
        self._report(progress, 45, f"分析標準音檔：{reference_path.name}")
        reference_segments = self._analyze_segments(
            reference_audio, reference_boundaries, cancel_event, None
        )
        if any(segment.error for segment in reference_segments):
            failed_indices = [str(segment.index + 1) for segment in reference_segments if segment.error]
            raise AnalysisError("標準音檔有無法分析的段落：" + ", ".join(failed_indices))

        total_candidates = sum(len(group) for group in prepared_groups)
        analyzed_candidates = 0
        all_results: list[list[FileAnalysisResult]] = []
        for config, prepared in zip(configs, prepared_groups):
            result_rows = [FileAnalysisResult(reference_path.name, segments=reference_segments)]
            folder_label = config.source_directory.name or str(config.source_directory)
            for item in prepared:
                self._check_cancel(cancel_event)
                if isinstance(item, FileAnalysisResult):
                    result_rows.append(item)
                else:
                    audio_path, audio, boundaries = item
                    percent = 45 + int(analyzed_candidates * 50 / max(1, total_candidates))
                    self._report(progress, percent, f"分析 [{folder_label}]：{audio_path.name}")
                    segments = self._analyze_segments(audio, boundaries, cancel_event, None)
                    for segment in segments:
                        reference_segment = reference_segments[segment.index]
                        if segment.error is None and reference_segment.error is None:
                            # Keep the reference vector private; Excel contains the requested top score only.
                            segment.distance = l2_distance(segment.scores, reference_segment.scores)
                    result_rows.append(FileAnalysisResult(audio_path.name, segments=segments))
                analyzed_candidates += 1

            self._write_excel(
                config.output_excel,
                result_rows,
                config.segment_count,
                {
                    "whisper_model": config.model_name,
                    "emotion_model": config.emotion_model_name,
                    "match_threshold": config.match_threshold,
                    "matching_strategy": "ordered_dp_with_context_relaxation",
                    "relaxed_match_floor": max(0.20, config.match_threshold - 0.05),
                    "segment_padding_seconds": config.segment_padding_seconds,
                    "noise_reduction_enabled": config.noise_reduction_enabled,
                    "noise_reduction_profile": config.noise_reduction_profile,
                },
            )
            all_results.append(result_rows)
            self._report(progress, 95, f"已輸出：{config.output_excel}")

        self._report(progress, 100, f"完成，共輸出 {len(configs)} 份 Excel")
        return all_results

    @staticmethod
    def _validate_shared_settings(configs: Sequence[AnalysisConfig]) -> None:
        primary = configs[0]
        shared = (
            primary.reference_audio.resolve(),
            primary.transcript_file.resolve(),
            primary.segment_count,
            primary.match_threshold,
            primary.model_name,
            primary.emotion_model_name,
            primary.segment_padding_seconds,
            primary.noise_reduction_enabled,
            primary.noise_reduction_profile,
        )
        for config in configs[1:]:
            candidate = (
                config.reference_audio.resolve(),
                config.transcript_file.resolve(),
                config.segment_count,
                config.match_threshold,
                config.model_name,
                config.emotion_model_name,
                config.segment_padding_seconds,
                config.noise_reduction_enabled,
                config.noise_reduction_profile,
            )
            if candidate != shared:
                raise AnalysisError("同一批次的標準音檔、文字分段與模型參數必須一致。")

    def _analyze_segments(
        self,
        audio: np.ndarray,
        boundaries: Sequence[SegmentedAudio],
        cancel_event: threading.Event | None,
        progress: ProgressCallback | None,
    ) -> list[SegmentResult]:
        assert self.emotion_classifier is not None
        results: list[SegmentResult] = []
        for index, boundary in enumerate(boundaries):
            self._check_cancel(cancel_event)
            percent = int(index * 100 / max(1, len(boundaries)))
            self._report(progress, percent, f"分析第 {index + 1} 段")
            start, end, confidence = boundary.start, boundary.end, boundary.confidence
            if not math.isfinite(start) or not math.isfinite(end):
                error = boundary.error or "分段失敗"
                results.append(
                    SegmentResult(index, start, end, confidence, np.zeros(9), "分段失敗", math.nan, error)
                )
                continue
            segment_audio = boundary.audio
            if segment_audio is None:
                start_index = max(0, int(start * 16000))
                end_index = min(len(audio), max(start_index + 1, int(end * 16000)))
                segment_audio = audio[start_index:end_index]
            if segment_audio is None or segment_audio.size == 0:
                results.append(
                    SegmentResult(index, start, end, confidence, np.zeros(9), "分段失敗", math.nan, boundary.error or "分段失敗")
                )
                continue
            try:
                vector, emotion, top_score = self.emotion_classifier.classify(segment_audio)
            except AnalysisError as exc:
                results.append(
                    SegmentResult(index, start, end, confidence, np.zeros(9), "分析失敗", math.nan, str(exc))
                )
                continue
            results.append(SegmentResult(index, start, end, confidence, vector, emotion, top_score))
        return results

    @staticmethod
    def _validate_config(config: AnalysisConfig) -> None:
        if config.segment_count <= 0:
            raise AnalysisError("分段數量必須大於 0。")
        if not 0.0 <= config.match_threshold <= 1.0:
            raise AnalysisError("Whisper 文字匹配門檻必須介於 0 與 1。")
        if not 0.0 <= config.segment_padding_seconds <= 5.0:
            raise AnalysisError("切段前後緩衝必須介於 0 與 5 秒。")
        if (
            config.noise_reduction_enabled
            and config.noise_reduction_profile not in NOISE_REDUCTION_FILTERS
        ):
            raise AnalysisError(f"不支援的去雜音模式：{config.noise_reduction_profile}")
        if not config.reference_audio.is_file():
            raise AnalysisError("找不到標準音檔。")
        if not config.transcript_file.is_file():
            raise AnalysisError("找不到分段文字檔。")
        if not config.source_directory.is_dir():
            raise AnalysisError("找不到待分析資料夾。")
        if config.reference_audio.suffix.casefold() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise AnalysisError("標準音檔格式不受支援。")
        if config.output_excel.suffix.casefold() != ".xlsx":
            raise AnalysisError("Excel 輸出檔案必須使用 .xlsx 副檔名。")

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event and cancel_event.is_set():
            raise AnalysisError("使用者已取消分析。")

    @staticmethod
    def _report(progress: ProgressCallback | None, value: int, message: str) -> None:
        if progress:
            progress(max(0, min(100, value)), message)

    @staticmethod
    def _write_excel(
        path: Path,
        rows: Sequence[FileAnalysisResult],
        segment_count: int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Write the stable legacy sheet plus analytical and raw-data sheets."""
        try:
            from .reporting import write_run_report
        except ImportError as exc:
            raise AnalysisError("Excel 輸出需要 openpyxl，請先安裝 requirements.txt。") from exc
        write_run_report(path, rows, segment_count, metadata)
