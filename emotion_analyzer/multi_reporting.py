from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .analytics import EMOTION_LABELS, RunAnalysis, normalize_subject_id
from .core import (
    AnalysisConfig,
    FileAnalysisResult,
    SegmentResult,
    enumerate_audio_files,
)
from .reporting import read_run_report, write_run_report


class MultiSetReportError(ValueError):
    """Raised when per-set reports cannot be combined safely."""


def _placeholder_segment(index: int, error: str) -> SegmentResult:
    return SegmentResult(
        index=index,
        start=math.nan,
        end=math.nan,
        confidence=0.0,
        scores=np.zeros(len(EMOTION_LABELS), dtype=np.float64),
        emotion="分析失敗",
        top_score=math.nan,
        error=error,
        distance=math.nan,
    )


def _observation_to_segment(observation, index: int) -> SegmentResult:
    scores = (
        np.asarray(observation.scores, dtype=np.float64)
        if observation.scores is not None
        else np.zeros(len(EMOTION_LABELS), dtype=np.float64)
    )
    return SegmentResult(
        index=index,
        start=observation.start if observation.start is not None else math.nan,
        end=observation.end if observation.end is not None else math.nan,
        confidence=(
            observation.confidence if observation.confidence is not None else 0.0
        ),
        scores=scores,
        emotion=observation.main_emotion or ("分析失敗" if observation.error else ""),
        top_score=(observation.top_score if observation.top_score is not None else math.nan),
        error=observation.error,
        distance=(
            observation.l2_distance
            if observation.l2_distance is not None
            else math.nan
        ),
    )


def _subject_map(run: RunAnalysis, set_name: str):
    subjects = {}
    duplicates: list[str] = []
    for subject in run.comparison_subjects:
        if subject.subject_id in subjects:
            duplicates.append(subject.subject_id)
        subjects[subject.subject_id] = subject
    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates)))
        raise MultiSetReportError(
            f"Analysis Set '{set_name}' 有重複受試者 ID：{duplicate_text}"
        )
    return subjects


def _reference_subject(run: RunAnalysis):
    return next((subject for subject in run.subjects if subject.is_reference), None)


def _segments_for_subject(
    subject,
    segment_count: int,
    global_offset: int,
    missing_error: str,
) -> list[SegmentResult]:
    if subject is None:
        return [
            _placeholder_segment(global_offset + local_index, missing_error)
            for local_index in range(segment_count)
        ]

    by_index = {segment.segment_index: segment for segment in subject.segments}
    results: list[SegmentResult] = []
    for local_index in range(1, segment_count + 1):
        global_index = global_offset + local_index - 1
        observation = by_index.get(local_index)
        if observation is None:
            results.append(
                _placeholder_segment(
                    global_index,
                    f"缺少第 {local_index} 段分析結果",
                )
            )
        else:
            results.append(_observation_to_segment(observation, global_index))
    return results


def build_flattened_rows(
    set_reports: Sequence[tuple[str, Path]],
) -> tuple[list[FileAnalysisResult], int, tuple[tuple[str, RunAnalysis], ...]]:
    """Flatten part1..N into the original single-run row/segment structure."""
    if not set_reports:
        raise MultiSetReportError("至少需要一份 Analysis Set 報表。")

    names = [name.strip() for name, _ in set_reports]
    if any(not name for name in names):
        raise MultiSetReportError("Analysis Set 名稱不可為空白。")
    if len({name.casefold() for name in names}) != len(names):
        raise MultiSetReportError("Analysis Set 名稱不可重複。")

    loaded: list[tuple[str, RunAnalysis]] = []
    subject_maps: list[dict[str, object]] = []
    subject_order: list[str] = []
    seen_subjects: set[str] = set()

    for set_name, report_path in set_reports:
        run = read_run_report(Path(report_path))
        loaded.append((set_name, run))
        subject_map = _subject_map(run, set_name)
        subject_maps.append(subject_map)
        for subject in run.comparison_subjects:
            if subject.subject_id not in seen_subjects:
                seen_subjects.add(subject.subject_id)
                subject_order.append(subject.subject_id)

    total_segment_count = sum(run.segment_count for _, run in loaded)
    if total_segment_count <= 0:
        raise MultiSetReportError("合併報表沒有任何可用段落。")

    reference_segments: list[SegmentResult] = []
    offset = 0
    for set_name, run in loaded:
        reference = _reference_subject(run)
        reference_segments.extend(
            _segments_for_subject(
                reference,
                run.segment_count,
                offset,
                f"{set_name}：缺少標準音檔分析結果",
            )
        )
        offset += run.segment_count

    rows: list[FileAnalysisResult] = [
        FileAnalysisResult("標準音檔", segments=reference_segments)
    ]

    for subject_id in subject_order:
        first_subject = next(
            (
                subject_map[subject_id]
                for subject_map in subject_maps
                if subject_id in subject_map
            ),
            None,
        )
        audio_name = (
            getattr(first_subject, "audio_name", None)
            or f"{subject_id}-combined.wav"
        )
        segments: list[SegmentResult] = []
        offset = 0
        for (set_name, run), subject_map in zip(loaded, subject_maps):
            subject = subject_map.get(subject_id)
            segments.extend(
                _segments_for_subject(
                    subject,
                    run.segment_count,
                    offset,
                    f"{set_name}：缺少來源音檔",
                )
            )
            offset += run.segment_count
        rows.append(FileAnalysisResult(str(audio_name), segments=segments))

    return rows, total_segment_count, tuple(loaded)


def write_failed_set_report(
    path: Path,
    config: AnalysisConfig,
    error: str,
) -> None:
    """Create a valid original-format report for a failed Set so the experiment can continue."""
    message = f"Analysis Set 分析失敗：{error}"
    reference_segments = [
        _placeholder_segment(index, message) for index in range(config.segment_count)
    ]
    rows: list[FileAnalysisResult] = [
        FileAnalysisResult(config.reference_audio.name or "標準音檔", segments=reference_segments)
    ]

    try:
        candidates = enumerate_audio_files(config.source_directory, config.recursive)
    except Exception:
        candidates = []
    reference = config.reference_audio.resolve()
    for audio_path in candidates:
        if audio_path.resolve() == reference:
            continue
        segments = [
            _placeholder_segment(index, message) for index in range(config.segment_count)
        ]
        rows.append(FileAnalysisResult(audio_path.name, segments=segments, error=message))

    write_run_report(
        Path(path),
        rows,
        config.segment_count,
        {
            "report_type": "failed_analysis_set",
            "set_error": error,
        },
    )


def write_multi_set_report(
    path: Path,
    set_reports: Sequence[tuple[str, Path]],
) -> RunAnalysis:
    """Write one workbook with exactly the original single-run report layout."""
    rows, total_segment_count, loaded = build_flattened_rows(set_reports)
    set_map_parts: list[str] = []
    offset = 1
    for set_name, run in loaded:
        end = offset + run.segment_count - 1
        set_map_parts.append(f"{set_name}:{offset}-{end}")
        offset = end + 1

    return write_run_report(
        Path(path),
        rows,
        total_segment_count,
        {
            "report_type": "multi_set_flattened_run_analysis",
            "analysis_set_count": len(loaded),
            "analysis_sets": " | ".join(name for name, _ in loaded),
            "analysis_set_segment_map": " | ".join(set_map_parts),
        },
    )
