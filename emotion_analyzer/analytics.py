from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from enum import Enum
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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
SCHEMA_VERSION = "2.0"


class AnalysisDataError(ValueError):
    """Raised when reports cannot be compared safely."""


class DistanceMetric(str, Enum):
    L1 = "l1"
    L2 = "l2"

    @property
    def display_name(self) -> str:
        return "L1 情緒截距" if self is DistanceMetric.L1 else "L2 情緒差距"


class PairingMode(str, Enum):
    NAME = "name"
    ORDER = "order"
    MANUAL = "manual"


@dataclass(frozen=True)
class SegmentObservation:
    subject_id: str
    audio_name: str
    is_reference: bool
    segment_index: int
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    main_emotion: str | None = None
    top_score: float | None = None
    scores: tuple[float, ...] | None = None
    top_three: tuple[tuple[str, float], ...] = ()
    l1_distance: float | None = None
    l2_distance: float | None = None
    max_change_emotion: str | None = None
    max_change_direction: str | None = None
    max_change_amount: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class SubjectSummary:
    subject_id: str
    audio_name: str
    is_reference: bool
    segments: tuple[SegmentObservation, ...]
    valid_segments: int
    mean_l1: float | None
    total_l1: float | None
    mean_l2: float | None
    total_l2: float | None
    max_l1_segment: int | None
    max_l2_segment: int | None
    error: str | None = None
    source_name: str | None = None
    record_id: str | None = None
    source_segment_count: int | None = None

    def metric_value(self, metric: DistanceMetric) -> float | None:
        return self.mean_l1 if metric is DistanceMetric.L1 else self.mean_l2

    @property
    def record_key(self) -> str:
        return self.record_id or self.subject_id


@dataclass(frozen=True)
class RunAnalysis:
    reference_audio: str
    subjects: tuple[SubjectSummary, ...]
    segment_count: int
    source_format: str
    schema_version: str = SCHEMA_VERSION
    source_files: tuple[str, ...] = ()

    @property
    def comparison_subjects(self) -> tuple[SubjectSummary, ...]:
        return tuple(subject for subject in self.subjects if not subject.is_reference)

    @property
    def available_metrics(self) -> frozenset[DistanceMetric]:
        metrics: set[DistanceMetric] = set()
        if any(subject.mean_l1 is not None for subject in self.comparison_subjects):
            metrics.add(DistanceMetric.L1)
        if any(subject.mean_l2 is not None for subject in self.comparison_subjects):
            metrics.add(DistanceMetric.L2)
        return frozenset(metrics)


@dataclass(frozen=True)
class ComparisonRow:
    first_subject: str | None
    second_subject: str | None
    first_audio: str | None
    second_audio: str | None
    first_value: float | None
    second_value: float | None
    group: str = "未分組"
    first_record_id: str | None = None
    second_record_id: str | None = None
    first_report: str | None = None
    second_report: str | None = None
    first_segment_values: tuple[tuple[int, float | None], ...] = ()
    second_segment_values: tuple[tuple[int, float | None], ...] = ()
    exclusion_reason: str | None = None

    @property
    def pairing_key(self) -> str:
        return self.first_record_id or self.second_record_id or self.first_subject or self.second_subject or ""

    @property
    def display_name(self) -> str:
        if self.first_subject and self.second_subject and self.first_subject != self.second_subject:
            return f"{self.first_subject} → {self.second_subject}"
        return self.first_subject or self.second_subject or ""


@dataclass(frozen=True)
class StatisticSummary:
    count: int
    mean: float | None
    standard_deviation: float | None


@dataclass(frozen=True)
class PairedTestResult:
    paired_count: int
    t_statistic: float | None
    p_value: float | None
    significant: bool | None
    note: str


@dataclass(frozen=True)
class ComparisonStatistics:
    name: str
    first: StatisticSummary
    second: StatisticSummary
    paired_test: PairedTestResult


@dataclass(frozen=True)
class GroupEffectTestResult:
    name: str
    experimental_count: int
    control_count: int
    experimental_mean_change: float | None
    control_mean_change: float | None
    effect_definition: str
    effect: float | None
    standard_error: float | None
    t_statistic: float | None
    degrees_freedom: float | None
    p_value: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None
    significant: bool | None
    note: str


@dataclass(frozen=True)
class ExperimentComparison:
    metric: DistanceMetric
    pairing_mode: PairingMode
    rows: tuple[ComparisonRow, ...]
    overall: ComparisonStatistics
    groups: tuple[ComparisonStatistics, ...]
    group_effect_tests: tuple[GroupEffectTestResult, ...]
    first_source: str = "第一次實驗"
    second_source: str = "第二次實驗"
    first_files: tuple[str, ...] = ()
    second_files: tuple[str, ...] = ()


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _subject_id(audio_name: str) -> str:
    stem = Path(audio_name).stem.strip()
    identifier = stem.split("-", 1)[0].strip()
    return identifier or stem or audio_name.strip()


def normalize_subject_id(value: str) -> str:
    """Use the student number before the first hyphen as the comparison ID."""
    return _subject_id(value)


def _summarize_subject(
    subject_id: str,
    audio_name: str,
    is_reference: bool,
    segments: Sequence[SegmentObservation],
    error: str | None = None,
) -> SubjectSummary:
    l1_values = [value for segment in segments if (value := segment.l1_distance) is not None]
    l2_values = [value for segment in segments if (value := segment.l2_distance) is not None]
    valid_segments = sum(1 for segment in segments if segment.error is None)
    max_l1 = max(
        (segment for segment in segments if segment.l1_distance is not None),
        key=lambda segment: segment.l1_distance or 0.0,
        default=None,
    )
    max_l2 = max(
        (segment for segment in segments if segment.l2_distance is not None),
        key=lambda segment: segment.l2_distance or 0.0,
        default=None,
    )
    return SubjectSummary(
        subject_id=subject_id,
        audio_name=audio_name,
        is_reference=is_reference,
        segments=tuple(sorted(segments, key=lambda segment: segment.segment_index)),
        valid_segments=valid_segments,
        mean_l1=statistics.fmean(l1_values) if l1_values else None,
        total_l1=sum(l1_values) if l1_values else None,
        mean_l2=statistics.fmean(l2_values) if l2_values else None,
        total_l2=sum(l2_values) if l2_values else None,
        max_l1_segment=max_l1.segment_index if max_l1 else None,
        max_l2_segment=max_l2.segment_index if max_l2 else None,
        error=error,
    )


def make_run_analysis(
    observations: Iterable[SegmentObservation],
    reference_audio: str,
    source_format: str,
    *,
    schema_version: str = SCHEMA_VERSION,
    subject_order: Sequence[tuple[str, str, bool, str | None]] | None = None,
) -> RunAnalysis:
    observations = tuple(observations)
    grouped: dict[str, list[SegmentObservation]] = {}
    discovered_order: list[tuple[str, str, bool, str | None]] = []
    for observation in observations:
        if observation.subject_id not in grouped:
            grouped[observation.subject_id] = []
            discovered_order.append(
                (observation.subject_id, observation.audio_name, observation.is_reference, None)
            )
        grouped[observation.subject_id].append(observation)

    ordered_subjects = list(subject_order or discovered_order)
    subjects = tuple(
        _summarize_subject(subject_id, audio_name, is_reference, grouped.get(subject_id, ()), error)
        for subject_id, audio_name, is_reference, error in ordered_subjects
    )
    segment_count = max(
        (observation.segment_index for observation in observations),
        default=0,
    )
    return RunAnalysis(reference_audio, subjects, segment_count, source_format, schema_version)


def merge_run_analyses(
    runs: Sequence[RunAnalysis],
    source_names: Sequence[str] | None = None,
) -> RunAnalysis:
    """Merge multiple report files into one comparison pool without losing provenance."""
    if not runs:
        raise AnalysisDataError("至少需要一份分析報表。")
    names = tuple(source_names or (f"報表 {index + 1}" for index in range(len(runs))))
    if len(names) != len(runs):
        raise AnalysisDataError("報表數量與來源名稱數量不一致。")

    subjects: list[SubjectSummary] = []
    for run_index, (run, source_name) in enumerate(zip(runs, names), 1):
        for subject_index, subject in enumerate(run.comparison_subjects, 1):
            subjects.append(
                replace(
                    subject,
                    is_reference=False,
                    source_name=source_name,
                    record_id=f"{run_index}:{subject_index}:{subject.subject_id}",
                    source_segment_count=run.segment_count,
                )
            )
    return RunAnalysis(
        reference_audio="",
        subjects=tuple(subjects),
        segment_count=max((run.segment_count for run in runs), default=0),
        source_format="pooled-reports",
        schema_version=SCHEMA_VERSION,
        source_files=names,
    )


def summarize_run(rows: Sequence[object], labels: Sequence[str] = EMOTION_LABELS) -> RunAnalysis:
    """Convert in-memory batch results into a complete, metric-neutral run analysis."""
    if not rows:
        raise AnalysisDataError("分析結果沒有任何音檔。")

    reference_row = rows[0]
    reference_name = str(getattr(reference_row, "audio_name", "標準音檔"))
    reference_vectors: dict[int, np.ndarray] = {}
    for segment in getattr(reference_row, "segments", ()):
        if getattr(segment, "error", None) is None:
            vector = np.asarray(getattr(segment, "scores", ()), dtype=np.float64)
            if vector.size == len(labels):
                reference_vectors[int(getattr(segment, "index")) + 1] = vector

    observations: list[SegmentObservation] = []
    subject_order: list[tuple[str, str, bool, str | None]] = []
    for row_index, row in enumerate(rows):
        audio_name = str(getattr(row, "audio_name", f"audio_{row_index + 1}"))
        subject_id = _subject_id(audio_name)
        is_reference = row_index == 0
        row_error = getattr(row, "error", None)
        subject_order.append((subject_id, audio_name, is_reference, row_error))
        for segment in getattr(row, "segments", ()):
            segment_index = int(getattr(segment, "index")) + 1
            error = getattr(segment, "error", None)
            vector = np.asarray(getattr(segment, "scores", ()), dtype=np.float64)
            vector_tuple: tuple[float, ...] | None = None
            top_three: tuple[tuple[str, float], ...] = ()
            l1_distance = None
            l2_distance = None
            max_emotion = None
            max_direction = None
            max_amount = None
            if error is None and vector.size == len(labels) and np.all(np.isfinite(vector)):
                vector_tuple = tuple(float(value) for value in vector)
                top_three = tuple(
                    sorted(zip(labels, vector_tuple), key=lambda pair: pair[1], reverse=True)[:3]
                )
                reference_vector = reference_vectors.get(segment_index)
                if reference_vector is not None:
                    delta = vector - reference_vector
                    l1_distance = float(np.sum(np.abs(delta)))
                    l2_distance = float(np.linalg.norm(delta))
                    max_index = int(np.argmax(np.abs(delta)))
                    max_amount = float(abs(delta[max_index]))
                    if max_amount > 0:
                        max_emotion = labels[max_index]
                        max_direction = "增加" if delta[max_index] > 0 else "減少"
            if l2_distance is None:
                l2_distance = _finite_number(getattr(segment, "distance", None))
            observations.append(
                SegmentObservation(
                    subject_id=subject_id,
                    audio_name=audio_name,
                    is_reference=is_reference,
                    segment_index=segment_index,
                    start=_finite_number(getattr(segment, "start", None)),
                    end=_finite_number(getattr(segment, "end", None)),
                    confidence=_finite_number(getattr(segment, "confidence", None)),
                    main_emotion=str(getattr(segment, "emotion", "")) or None,
                    top_score=_finite_number(getattr(segment, "top_score", None)),
                    scores=vector_tuple,
                    top_three=top_three,
                    l1_distance=l1_distance,
                    l2_distance=l2_distance,
                    max_change_emotion=max_emotion,
                    max_change_direction=max_direction,
                    max_change_amount=max_amount,
                    error=error,
                )
            )

    return make_run_analysis(
        observations,
        reference_name,
        "oeeana-v2",
        subject_order=subject_order,
    )


def _unique_subjects(subjects: Sequence[SubjectSummary], report_name: str) -> dict[str, SubjectSummary]:
    result: dict[str, SubjectSummary] = {}
    duplicates: set[str] = set()
    for subject in subjects:
        if subject.subject_id in result:
            duplicates.add(subject.subject_id)
        result[subject.subject_id] = subject
    if duplicates:
        raise AnalysisDataError(f"{report_name}有重複受試者名稱：{', '.join(sorted(duplicates))}")
    return result


def create_pairing_rows(
    first: RunAnalysis,
    second: RunAnalysis,
    mode: PairingMode,
    manual_pairs: Sequence[tuple[str, str]] | None = None,
) -> tuple[tuple[SubjectSummary | None, SubjectSummary | None], ...]:
    first_subjects = first.comparison_subjects
    second_subjects = second.comparison_subjects
    if mode is PairingMode.ORDER:
        return tuple(zip_longest(first_subjects, second_subjects))

    if mode is PairingMode.NAME:
        first_by_id = _unique_subjects(first_subjects, "前測資料池")
        second_by_id = _unique_subjects(second_subjects, "後測資料池")
        keys = list(first_by_id)
        keys.extend(key for key in second_by_id if key not in first_by_id)
        return tuple((first_by_id.get(key), second_by_id.get(key)) for key in keys)

    first_by_id = {subject.record_key: subject for subject in first_subjects}
    second_by_id = {subject.record_key: subject for subject in second_subjects}
    pairs = list(manual_pairs or ())
    first_used: set[str] = set()
    second_used: set[str] = set()
    result: list[tuple[SubjectSummary | None, SubjectSummary | None]] = []
    for first_id, second_id in pairs:
        if first_id not in first_by_id or second_id not in second_by_id:
            raise AnalysisDataError(f"手動配對不存在：{first_id} → {second_id}")
        if first_id in first_used or second_id in second_used:
            raise AnalysisDataError("手動配對必須是一對一，不可重複使用受試者。")
        first_used.add(first_id)
        second_used.add(second_id)
        result.append((first_by_id[first_id], second_by_id[second_id]))
    result.extend((subject, None) for subject in first_subjects if subject.record_key not in first_used)
    result.extend((None, subject) for subject in second_subjects if subject.record_key not in second_used)
    return tuple(result)


def _segment_metric_values(
    subject: SubjectSummary | None,
    metric: DistanceMetric,
) -> tuple[tuple[int, float | None], ...]:
    if subject is None:
        return ()
    return tuple(
        (
            segment.segment_index,
            segment.l1_distance if metric is DistanceMetric.L1 else segment.l2_distance,
        )
        for segment in sorted(subject.segments, key=lambda item: item.segment_index)
    )


def _summary(values: Iterable[float | None]) -> StatisticSummary:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return StatisticSummary(
        count=len(clean),
        mean=statistics.fmean(clean) if clean else None,
        standard_deviation=statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None),
    )


def _paired_test(rows: Sequence[ComparisonRow]) -> PairedTestResult:
    paired = [row for row in rows if row.first_value is not None and row.second_value is not None]
    if len(paired) < 2:
        return PairedTestResult(len(paired), None, None, None, "有效成對樣本數不足，至少需要 2 對。")
    first_values = [row.first_value for row in paired]
    second_values = [row.second_value for row in paired]
    try:
        from scipy.stats import ttest_rel

        result = ttest_rel(first_values, second_values)
        t_statistic = _finite_number(result.statistic)
        p_value = _finite_number(result.pvalue)
    except ImportError as exc:
        raise AnalysisDataError("雙實驗統計需要 scipy，請先安裝 requirements.txt。") from exc
    if t_statistic is None or p_value is None:
        return PairedTestResult(len(paired), None, None, None, "成對差值沒有可估計的變異，無法計算 t-test。")
    return PairedTestResult(
        len(paired),
        t_statistic,
        p_value,
        p_value < 0.05,
        "達顯著差異（p < 0.05）" if p_value < 0.05 else "未達顯著差異（p ≥ 0.05）",
    )


def _comparison_statistics(name: str, rows: Sequence[ComparisonRow]) -> ComparisonStatistics:
    return ComparisonStatistics(
        name,
        _summary(row.first_value for row in rows),
        _summary(row.second_value for row in rows),
        _paired_test(rows),
    )


def _paired_group_rows(rows: Sequence[ComparisonRow], group: str) -> list[ComparisonRow]:
    return [
        row
        for row in rows
        if row.group == group
        and row.first_value is not None
        and row.second_value is not None
        and math.isfinite(row.first_value)
        and math.isfinite(row.second_value)
    ]


def _group_change_context(
    rows: Sequence[ComparisonRow],
) -> tuple[list[ComparisonRow], list[ComparisonRow], list[float], list[float]]:
    experimental = _paired_group_rows(rows, "實驗組")
    control = _paired_group_rows(rows, "對照組")
    experimental_changes = [row.second_value - row.first_value for row in experimental]
    control_changes = [row.second_value - row.first_value for row in control]
    return experimental, control, experimental_changes, control_changes


def _unavailable_group_test(
    name: str,
    experimental_changes: Sequence[float],
    control_changes: Sequence[float],
    effect_definition: str,
    note: str,
) -> GroupEffectTestResult:
    return GroupEffectTestResult(
        name=name,
        experimental_count=len(experimental_changes),
        control_count=len(control_changes),
        experimental_mean_change=(
            statistics.fmean(experimental_changes) if experimental_changes else None
        ),
        control_mean_change=statistics.fmean(control_changes) if control_changes else None,
        effect_definition=effect_definition,
        effect=None,
        standard_error=None,
        t_statistic=None,
        degrees_freedom=None,
        p_value=None,
        confidence_interval_low=None,
        confidence_interval_high=None,
        significant=None,
        note=note,
    )


def _welch_change_test(rows: Sequence[ComparisonRow]) -> GroupEffectTestResult:
    _, _, experimental_changes, control_changes = _group_change_context(rows)
    effect_definition = "平均變化量差（實驗組－對照組）"
    if len(experimental_changes) < 2 or len(control_changes) < 2:
        return _unavailable_group_test(
            "Welch t 檢定",
            experimental_changes,
            control_changes,
            effect_definition,
            "實驗組與對照組各至少需要 2 筆有效成對資料。",
        )

    experimental_variance = statistics.variance(experimental_changes)
    control_variance = statistics.variance(control_changes)
    experimental_component = experimental_variance / len(experimental_changes)
    control_component = control_variance / len(control_changes)
    standard_error_squared = experimental_component + control_component
    if standard_error_squared <= 0:
        return _unavailable_group_test(
            "Welch t 檢定",
            experimental_changes,
            control_changes,
            effect_definition,
            "兩組變化量沒有可估計的變異，無法計算 Welch t 檢定。",
        )

    effect = statistics.fmean(experimental_changes) - statistics.fmean(control_changes)
    standard_error = math.sqrt(standard_error_squared)
    denominator = (
        experimental_component**2 / (len(experimental_changes) - 1)
        + control_component**2 / (len(control_changes) - 1)
    )
    degrees_freedom = standard_error_squared**2 / denominator
    t_statistic = effect / standard_error
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:
        raise AnalysisDataError("組間統計需要 scipy，請先安裝 requirements.txt。") from exc
    p_value = float(2 * student_t.sf(abs(t_statistic), degrees_freedom))
    critical_value = float(student_t.ppf(0.975, degrees_freedom))
    confidence_low = effect - critical_value * standard_error
    confidence_high = effect + critical_value * standard_error
    significant = p_value < 0.05
    if significant:
        direction = "實驗組下降幅度較大" if effect < 0 else "對照組下降幅度較大"
        note = f"達顯著組間差異（p < 0.05）；{direction}。"
    else:
        note = "未達顯著組間差異（p ≥ 0.05）。"
    return GroupEffectTestResult(
        name="Welch t 檢定",
        experimental_count=len(experimental_changes),
        control_count=len(control_changes),
        experimental_mean_change=statistics.fmean(experimental_changes),
        control_mean_change=statistics.fmean(control_changes),
        effect_definition=effect_definition,
        effect=effect,
        standard_error=standard_error,
        t_statistic=t_statistic,
        degrees_freedom=degrees_freedom,
        p_value=p_value,
        confidence_interval_low=confidence_low,
        confidence_interval_high=confidence_high,
        significant=significant,
        note=note,
    )


def _ancova_test(rows: Sequence[ComparisonRow]) -> GroupEffectTestResult:
    experimental, control, experimental_changes, control_changes = _group_change_context(rows)
    effect_definition = "校正前測後的後測差（實驗組－對照組）"
    if len(experimental) < 2 or len(control) < 2:
        return _unavailable_group_test(
            "ANCOVA",
            experimental_changes,
            control_changes,
            effect_definition,
            "實驗組與對照組各至少需要 2 筆有效成對資料。",
        )

    paired_rows = [*experimental, *control]
    outcomes = np.asarray([row.second_value for row in paired_rows], dtype=float)
    baselines = np.asarray([row.first_value for row in paired_rows], dtype=float)
    group_indicator = np.asarray(
        [1.0 if row.group == "實驗組" else 0.0 for row in paired_rows], dtype=float
    )
    design = np.column_stack((np.ones(len(paired_rows)), baselines, group_indicator))
    parameter_count = design.shape[1]
    residual_degrees_freedom = len(paired_rows) - parameter_count
    if residual_degrees_freedom <= 0 or np.linalg.matrix_rank(design) < parameter_count:
        return _unavailable_group_test(
            "ANCOVA",
            experimental_changes,
            control_changes,
            effect_definition,
            "資料不足或設計矩陣無法估計前測與組別的獨立效果。",
        )

    coefficients, _, _, _ = np.linalg.lstsq(design, outcomes, rcond=None)
    residuals = outcomes - design @ coefficients
    residual_variance = float(residuals @ residuals / residual_degrees_freedom)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    standard_error = _finite_number(math.sqrt(max(float(covariance[2, 2]), 0.0)))
    effect = _finite_number(coefficients[2])
    if effect is None or standard_error is None or standard_error <= 0:
        return _unavailable_group_test(
            "ANCOVA",
            experimental_changes,
            control_changes,
            effect_definition,
            "模型殘差沒有可估計的變異，無法計算 ANCOVA 組別效果。",
        )

    t_statistic = effect / standard_error
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:
        raise AnalysisDataError("組間統計需要 scipy，請先安裝 requirements.txt。") from exc
    p_value = float(2 * student_t.sf(abs(t_statistic), residual_degrees_freedom))
    critical_value = float(student_t.ppf(0.975, residual_degrees_freedom))
    confidence_low = effect - critical_value * standard_error
    confidence_high = effect + critical_value * standard_error
    significant = p_value < 0.05
    if significant:
        direction = "實驗組校正後後測較低" if effect < 0 else "實驗組校正後後測較高"
        note = f"達顯著組間差異（p < 0.05）；{direction}。"
    else:
        note = "校正前測後未達顯著組間差異（p ≥ 0.05）。"
    return GroupEffectTestResult(
        name="ANCOVA",
        experimental_count=len(experimental),
        control_count=len(control),
        experimental_mean_change=statistics.fmean(experimental_changes),
        control_mean_change=statistics.fmean(control_changes),
        effect_definition=effect_definition,
        effect=effect,
        standard_error=standard_error,
        t_statistic=t_statistic,
        degrees_freedom=float(residual_degrees_freedom),
        p_value=p_value,
        confidence_interval_low=confidence_low,
        confidence_interval_high=confidence_high,
        significant=significant,
        note=note,
    )


def compare_runs(
    first: RunAnalysis,
    second: RunAnalysis,
    *,
    metric: DistanceMetric = DistanceMetric.L1,
    pairing_mode: PairingMode = PairingMode.NAME,
    manual_pairs: Sequence[tuple[str, str]] | None = None,
    groups: Mapping[str, str] | None = None,
    exclusion_reasons: Mapping[str, str] | None = None,
    first_source: str = "第一次實驗",
    second_source: str = "第二次實驗",
) -> ExperimentComparison:
    """Pair two run reports and calculate overall/group descriptive and paired statistics."""
    groups = groups or {}
    exclusion_reasons = exclusion_reasons or {}
    pairings = create_pairing_rows(first, second, pairing_mode, manual_pairs)
    rows: list[ComparisonRow] = []
    for first_subject, second_subject in pairings:
        pairing_key = first_subject.record_key if first_subject else second_subject.record_key
        rows.append(
            ComparisonRow(
                first_subject=first_subject.subject_id if first_subject else None,
                second_subject=second_subject.subject_id if second_subject else None,
                first_audio=first_subject.audio_name if first_subject else None,
                second_audio=second_subject.audio_name if second_subject else None,
                first_value=first_subject.metric_value(metric) if first_subject else None,
                second_value=second_subject.metric_value(metric) if second_subject else None,
                group=groups.get(pairing_key, "未分組"),
                first_record_id=first_subject.record_key if first_subject else None,
                second_record_id=second_subject.record_key if second_subject else None,
                first_report=first_subject.source_name if first_subject else None,
                second_report=second_subject.source_name if second_subject else None,
                first_segment_values=_segment_metric_values(first_subject, metric),
                second_segment_values=_segment_metric_values(second_subject, metric),
                exclusion_reason=exclusion_reasons.get(pairing_key),
            )
        )

    included_rows = [row for row in rows if row.group != "排除"]
    if not any(row.first_value is not None for row in included_rows):
        raise AnalysisDataError(f"第一次實驗不包含可用的 {metric.display_name}。")
    if not any(row.second_value is not None for row in included_rows):
        raise AnalysisDataError(f"第二次實驗不包含可用的 {metric.display_name}。")

    overall = _comparison_statistics("整體", included_rows)
    group_results = tuple(
        _comparison_statistics(group_name, [row for row in rows if row.group == group_name])
        for group_name in ("實驗組", "對照組")
    )
    group_effect_tests = (_welch_change_test(rows), _ancova_test(rows))
    return ExperimentComparison(
        metric,
        pairing_mode,
        tuple(rows),
        overall,
        group_results,
        group_effect_tests,
        first_source,
        second_source,
        first.source_files,
        second.source_files,
    )
