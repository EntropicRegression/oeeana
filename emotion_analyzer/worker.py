from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .core import (
    AnalysisConfig,
    BatchAnalyzer,
    DEFAULT_EMOTION_MODEL,
    DEFAULT_NOISE_REDUCTION_PROFILE,
    PreprocessResult,
)
from .multi_reporting import write_failed_set_report, write_multi_set_report


EVENT_PREFIX = "@@OEEANA_EVENT@@"
EventSink = Callable[[dict[str, object]], None]
JOB_PHASES = frozenset({"full", "preprocess", "analyze", "single_cut"})


def _config_to_payload(config: AnalysisConfig) -> dict[str, object]:
    return {
        "reference_audio": str(config.reference_audio),
        "segment_count": config.segment_count,
        "transcript_file": str(config.transcript_file),
        "source_directory": str(config.source_directory),
        "match_threshold": config.match_threshold,
        "output_excel": str(config.output_excel),
        "model_name": config.model_name,
        "emotion_model_name": config.emotion_model_name,
        "segment_padding_seconds": config.segment_padding_seconds,
        "noise_reduction_enabled": config.noise_reduction_enabled,
        "noise_reduction_profile": config.noise_reduction_profile,
        "recursive": config.recursive,
    }


def create_job_file(
    config: AnalysisConfig | Sequence[AnalysisConfig],
    *,
    phase: str = "full",
    skip_missing: bool = False,
) -> Path:
    """Serialize one or more analysis requests for the isolated model process."""
    if phase not in JOB_PHASES:
        raise ValueError(f"不支援的工作階段：{phase}")
    if isinstance(config, AnalysisConfig):
        payload: dict[str, object] = _config_to_payload(config)
        payload["phase"] = phase
    else:
        configs = list(config)
        if not configs:
            raise ValueError("批次工作至少需要一個分析設定。")
        payload = {
            "phase": phase,
            "configs": [_config_to_payload(item) for item in configs],
        }
    if skip_missing:
        payload["skip_missing"] = True
    descriptor, raw_path = tempfile.mkstemp(prefix="oeeana-job-", suffix=".json")
    os.close(descriptor)
    path = Path(raw_path)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def create_single_cut_job_file(
    audio_path: Path,
    transcript_file: Path,
    *,
    match_threshold: float = 0.30,
    segment_padding_seconds: float = 1.0,
    noise_reduction_enabled: bool = True,
) -> Path:
    """Serialize a cut-only request that targets exactly one audio file."""
    descriptor, raw_path = tempfile.mkstemp(prefix="oeeana-single-cut-", suffix=".json")
    os.close(descriptor)
    path = Path(raw_path)
    path.write_text(
        json.dumps(
            {
                "phase": "single_cut",
                "audio_path": str(audio_path),
                "transcript_file": str(transcript_file),
                "match_threshold": match_threshold,
                "segment_padding_seconds": segment_padding_seconds,
                "noise_reduction_enabled": noise_reduction_enabled,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _payload_to_config(payload: Mapping[str, object]) -> AnalysisConfig:
    return AnalysisConfig(
        reference_audio=Path(str(payload["reference_audio"])),
        segment_count=int(payload["segment_count"]),
        transcript_file=Path(str(payload["transcript_file"])),
        source_directory=Path(str(payload["source_directory"])),
        match_threshold=float(payload["match_threshold"]),
        output_excel=Path(str(payload["output_excel"])),
        model_name=str(payload.get("model_name", "large-v3")),
        emotion_model_name=str(payload.get("emotion_model_name", DEFAULT_EMOTION_MODEL)),
        segment_padding_seconds=float(payload.get("segment_padding_seconds", 1.0)),
        noise_reduction_enabled=bool(payload.get("noise_reduction_enabled", True)),
        noise_reduction_profile=str(
            payload.get("noise_reduction_profile", DEFAULT_NOISE_REDUCTION_PROFILE)
        ),
        recursive=bool(payload.get("recursive", False)),
    )


def load_job_files(path: Path) -> tuple[AnalysisConfig, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("分析工作檔格式錯誤。")
    raw_configs = payload.get("configs")
    if raw_configs is None:
        return (_payload_to_config(payload),)
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("批次分析工作沒有任何資料夾。")
    if not all(isinstance(item, dict) for item in raw_configs):
        raise ValueError("批次分析工作格式錯誤。")
    return tuple(_payload_to_config(item) for item in raw_configs)


def load_job_phase(path: Path) -> str:
    """Load the requested phase while accepting legacy jobs as full runs."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("分析工作檔格式錯誤。")
    phase = str(payload.get("phase", payload.get("mode", "full")))
    if phase not in JOB_PHASES:
        raise ValueError(f"不支援的工作階段：{phase}")
    return phase


def load_job_skip_missing(path: Path) -> bool:
    """Load the skip_missing flag from the job file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("skip_missing", False))


def load_job_file(path: Path) -> AnalysisConfig:
    """Load the legacy single-config job format."""
    configs = load_job_files(path)
    if len(configs) != 1:
        raise ValueError("此工作檔包含多個分析資料夾，請使用 load_job_files。")
    return configs[0]


def encode_event(event: Mapping[str, object]) -> str:
    return EVENT_PREFIX + json.dumps(dict(event), ensure_ascii=False, separators=(",", ":"))


def decode_event(line: str) -> dict[str, object] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        event = json.loads(line[len(EVENT_PREFIX) :])
    except (json.JSONDecodeError, TypeError):
        return None
    return event if isinstance(event, dict) and isinstance(event.get("event"), str) else None


def _batch_compatible(configs: Sequence[AnalysisConfig]) -> bool:
    """Return True when the optimized shared-reference BatchAnalyzer path is valid."""
    if len(configs) <= 1:
        return True
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
    return all(
        (
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
        == shared
        for config in configs[1:]
    )


def _scaled_progress(
    emit: EventSink,
    index: int,
    total: int,
    set_label: str,
) -> Callable[[int, str], None]:
    start = index * 100 / total
    width = 100 / total

    def report(value: int, message: str) -> None:
        overall = int(start + max(0, min(100, value)) * width / 100)
        emit(
            {
                "event": "progress",
                "value": overall,
                "message": f"[{set_label}] {message}",
            }
        )

    return report


def _set_name_from_output(path: Path, index: int) -> str:
    stem = path.stem
    prefix = "emotion_analysis_result_"
    if stem.casefold().startswith(prefix.casefold()):
        name = stem[len(prefix) :].strip()
        if name:
            return name
    return f"part{index + 1}"


def _combined_output_path(configs: Sequence[AnalysisConfig]) -> Path:
    parents = [str(config.output_excel.resolve().parent) for config in configs]
    try:
        common = Path(os.path.commonpath(parents))
    except ValueError:
        common = configs[0].output_excel.resolve().parent
    if common.exists() and common.is_file():
        common = configs[0].output_excel.resolve().parent
    return common / "emotion_analysis_result.xlsx"


def _write_combined_report(configs: Sequence[AnalysisConfig]) -> Path:
    set_reports = [
        (_set_name_from_output(config.output_excel, index), config.output_excel)
        for index, config in enumerate(configs)
    ]
    output = _combined_output_path(configs)
    write_multi_set_report(output, set_reports)
    return output


def _remove_set_reports(configs: Sequence[AnalysisConfig], combined_output: Path) -> None:
    combined_key = str(combined_output.resolve()).casefold()
    for config in configs:
        path = config.output_excel
        try:
            if str(path.resolve()).casefold() != combined_key:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def run_job(
    job_file: Path,
    *,
    analyzer_factory: Callable[[], BatchAnalyzer] = BatchAnalyzer,
    event_sink: EventSink | None = None,
) -> int:
    """Run one complete analysis behind a small progress-event interface."""

    def emit(event: dict[str, object]) -> None:
        if event_sink is not None:
            event_sink(event)
        else:
            print(encode_event(event), flush=True)

    combined_output: Path | None = None
    set_errors: list[str] = []
    try:
        phase = load_job_phase(job_file)
        if phase == "single_cut":
            payload = json.loads(job_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("單音檔切段工作格式錯誤。")
            audio_path = Path(str(payload.get("audio_path", "")))
            transcript_file = Path(str(payload.get("transcript_file", "")))
            analyzer = analyzer_factory()
            result = analyzer.cut_single_audio(
                audio_path,
                transcript_file,
                lambda value, message: emit(
                    {"event": "progress", "value": int(value), "message": str(message)}
                ),
                match_threshold=float(payload.get("match_threshold", 0.30)),
                segment_padding_seconds=float(
                    payload.get("segment_padding_seconds", 1.0)
                ),
                noise_reduction_enabled=bool(
                    payload.get("noise_reduction_enabled", True)
                ),
            )
            emit(
                {
                    "event": "finished",
                    "phase": phase,
                    "chopped_paths": [str(path) for path in result.chopped_paths],
                    "missing_chopped_paths": [str(path) for path in result.missing_paths],
                    "errors": list(result.errors),
                }
            )
            return 0

        configs = load_job_files(job_file)
        requested_skip_missing = load_job_skip_missing(job_file)
        compatible = _batch_compatible(configs)
        multi_set = len(configs) > 1 and not compatible
        # Multi-set experiments are tolerant by design. Missing chopped audio becomes
        # an error cell in the final report instead of aborting the whole experiment.
        skip_missing = requested_skip_missing or multi_set

        if compatible:
            analyzer = analyzer_factory()
            report_progress = lambda value, message: emit(
                {"event": "progress", "value": int(value), "message": str(message)}
            )
            if phase == "preprocess":
                result = analyzer.preprocess_many(configs, report_progress)
            elif phase == "analyze" and len(configs) == 1:
                analyzer.analyze_chopped(
                    configs[0], report_progress, skip_missing=skip_missing
                )
            elif phase == "analyze":
                analyzer.analyze_chopped_many(
                    configs, report_progress, skip_missing=skip_missing
                )
            elif len(configs) == 1:
                analyzer.run(configs[0], report_progress)
            else:
                analyzer.run_many(configs, report_progress, skip_missing=skip_missing)
        else:
            source_files: list[Path] = []
            chopped_paths: list[Path] = []
            missing_paths: list[Path] = []
            preprocess_errors: list[str] = []
            total = len(configs)

            for index, config in enumerate(configs):
                label = _set_name_from_output(config.output_excel, index)
                report_progress = _scaled_progress(emit, index, total, label)
                analyzer = analyzer_factory()
                if phase == "preprocess":
                    partial = analyzer.preprocess(config, report_progress)
                    source_files.extend(partial.source_files)
                    chopped_paths.extend(partial.chopped_paths)
                    missing_paths.extend(partial.missing_paths)
                    preprocess_errors.extend(partial.errors)
                    continue

                try:
                    if phase == "analyze":
                        analyzer.analyze_chopped(
                            config,
                            report_progress,
                            skip_missing=True,
                        )
                    else:
                        analyzer.run(
                            config,
                            report_progress,
                        )
                except Exception as exc:
                    # One damaged/missing Set must not prevent the rest of the experiment.
                    message = f"[{label}] {exc}"
                    set_errors.append(message)
                    emit(
                        {
                            "event": "progress",
                            "value": int((index + 1) * 100 / total),
                            "message": message + "；已記錄並繼續。",
                        }
                    )
                    write_failed_set_report(config.output_excel, config, str(exc))

            if phase == "preprocess":
                result = PreprocessResult(
                    tuple(source_files),
                    tuple(chopped_paths),
                    tuple(missing_paths),
                    tuple(preprocess_errors),
                )

        if phase != "preprocess" and multi_set:
            emit(
                {
                    "event": "progress",
                    "value": 99,
                    "message": "彙整單一實驗報表...",
                }
            )
            combined_output = _write_combined_report(configs)
            _remove_set_reports(configs, combined_output)
    except Exception as exc:
        emit({"event": "failed", "message": str(exc)})
        return 1

    if phase == "preprocess":
        emit(
            {
                "event": "finished",
                "phase": phase,
                "chopped_paths": [str(path) for path in result.chopped_paths],
                "missing_chopped_paths": [str(path) for path in result.missing_paths],
                "errors": list(result.errors),
            }
        )
        return 0

    if combined_output is not None:
        event: dict[str, object] = {
            "event": "finished",
            "phase": phase,
            "output": str(combined_output),
        }
        if set_errors:
            event["errors"] = set_errors
        emit(event)
        return 0

    outputs = [str(config.output_excel) for config in configs]
    event = {"event": "finished", "phase": phase}
    if len(outputs) == 1:
        event["output"] = outputs[0]
    else:
        event["outputs"] = outputs
    emit(event)
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: python -m emotion_analyzer.worker <job.json>", file=sys.stderr)
        return 2
    return run_job(Path(arguments[0]))


if __name__ == "__main__":
    raise SystemExit(main())
