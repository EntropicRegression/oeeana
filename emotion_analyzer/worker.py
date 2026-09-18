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
)


EVENT_PREFIX = "@@OEEANA_EVENT@@"
EventSink = Callable[[dict[str, object]], None]
JOB_PHASES = frozenset({"full", "preprocess", "analyze"})


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
    descriptor, raw_path = tempfile.mkstemp(prefix="oeeana-job-", suffix=".json")
    os.close(descriptor)
    path = Path(raw_path)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
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

    try:
        configs = load_job_files(job_file)
        phase = load_job_phase(job_file)
        analyzer = analyzer_factory()
        report_progress = lambda value, message: emit(
            {"event": "progress", "value": int(value), "message": str(message)}
        )
        if phase == "preprocess":
            result = analyzer.preprocess_many(configs, report_progress)
        elif phase == "analyze" and len(configs) == 1:
            analyzer.analyze_chopped(configs[0], report_progress)
        elif phase == "analyze":
            analyzer.analyze_chopped_many(configs, report_progress)
        elif len(configs) == 1:
            analyzer.run(configs[0], report_progress)
        else:
            analyzer.run_many(configs, report_progress)
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

    outputs = [str(config.output_excel) for config in configs]
    if phase == "analyze" and len(outputs) == 1:
        emit({"event": "finished", "phase": phase, "output": outputs[0]})
    elif phase == "analyze":
        emit({"event": "finished", "phase": phase, "outputs": outputs})
    elif len(outputs) == 1:
        emit({"event": "finished", "output": outputs[0]})
    else:
        emit({"event": "finished", "outputs": outputs})
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
