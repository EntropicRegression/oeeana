from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .core import (
    AnalysisConfig,
    DEFAULT_EMOTION_MODEL,
    EMOTION_MODEL_OPTIONS,
    read_transcript,
)
from .worker import create_job_file, decode_event


@dataclass(frozen=True)
class AnalysisSet:
    name: str
    source_directory: Path
    reference_audio: Path
    transcript_file: Path
    segment_count: int
    output_excel: Path


def _safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "set"


def run_multi_gui() -> int:
    try:
        from PySide6.QtCore import QProcess
        from PySide6.QtWidgets import (
            QApplication,
            QAbstractItemView,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QHBoxLayout,
            QHeaderView,
            QInputDialog,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print("缺少 PySide6，請先安裝 requirements.txt。", file=sys.stderr)
        return 1

    class MultiSetWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("OEEANA 多段 Analysis Set")
            self.resize(1280, 760)
            self.entries: list[AnalysisSet] = []
            self.process: QProcess | None = None
            self.job_file: Path | None = None
            self.stdout_buffer = ""

            root = QWidget()
            layout = QVBoxLayout(root)

            hint = QLabel(
                "每個 Analysis Set 可使用自己的標準音檔與分段文字。"
                "同一 Set 的來源資料夾只放該段錄音，例如 pretest/part1、pretest/part2。"
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)

            self.table = QTableWidget(0, 6)
            self.table.setHorizontalHeaderLabels(
                ["Set", "來源資料夾", "標準音檔", "文字檔", "段落數", "輸出"]
            )
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.table, 2)

            set_buttons = QHBoxLayout()
            add_button = QPushButton("新增 Analysis Set")
            add_button.clicked.connect(self.add_set)
            remove_button = QPushButton("移除選取")
            remove_button.clicked.connect(self.remove_selected)
            clear_button = QPushButton("全部清除")
            clear_button.clicked.connect(self.clear_sets)
            set_buttons.addWidget(add_button)
            set_buttons.addWidget(remove_button)
            set_buttons.addWidget(clear_button)
            set_buttons.addStretch(1)
            layout.addLayout(set_buttons)

            params = QHBoxLayout()
            params.addWidget(QLabel("Whisper 門檻"))
            self.threshold = QDoubleSpinBox()
            self.threshold.setRange(0.0, 1.0)
            self.threshold.setDecimals(2)
            self.threshold.setSingleStep(0.05)
            self.threshold.setValue(0.30)
            params.addWidget(self.threshold)

            params.addWidget(QLabel("切段前後緩衝"))
            self.padding = QDoubleSpinBox()
            self.padding.setRange(0.0, 5.0)
            self.padding.setDecimals(1)
            self.padding.setSingleStep(0.1)
            self.padding.setValue(1.0)
            self.padding.setSuffix(" 秒")
            params.addWidget(self.padding)

            params.addWidget(QLabel("emotion2vec+"))
            self.model = QComboBox()
            for label, model_name in EMOTION_MODEL_OPTIONS:
                self.model.addItem(label, model_name)
            self.model.setCurrentIndex(max(0, self.model.findData(DEFAULT_EMOTION_MODEL)))
            params.addWidget(self.model, 1)

            self.noise_reduction = QCheckBox("背景雜音處理")
            self.noise_reduction.setChecked(True)
            params.addWidget(self.noise_reduction)
            layout.addLayout(params)

            action_row = QHBoxLayout()
            self.preprocess_button = QPushButton("1. 全部切段")
            self.preprocess_button.clicked.connect(lambda: self.start_job("preprocess"))
            self.analyze_button = QPushButton("2. 全部情緒分析")
            self.analyze_button.clicked.connect(lambda: self.start_job("analyze"))
            self.cancel_button = QPushButton("取消")
            self.cancel_button.clicked.connect(self.cancel_job)
            self.cancel_button.setEnabled(False)
            action_row.addWidget(self.preprocess_button)
            action_row.addWidget(self.analyze_button)
            action_row.addWidget(self.cancel_button)
            action_row.addStretch(1)
            layout.addLayout(action_row)

            self.progress = QProgressBar()
            self.status = QLabel("請新增至少一個 Analysis Set。")
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.progress)
            layout.addWidget(self.status)
            layout.addWidget(self.log, 1)

            self.setCentralWidget(root)

        def add_set(self):
            name, ok = QInputDialog.getText(
                self, "Analysis Set", "Set 名稱，例如 part1 / 問題1："
            )
            if not ok or not name.strip():
                return

            source = QFileDialog.getExistingDirectory(
                self, f"[{name}] 選擇來源錄音資料夾"
            )
            if not source:
                return

            reference, _ = QFileDialog.getOpenFileName(
                self,
                f"[{name}] 選擇標準音檔",
                "",
                "Audio (*.wav *.mp3 *.m4a *.flac)",
            )
            if not reference:
                return

            transcript, _ = QFileDialog.getOpenFileName(
                self,
                f"[{name}] 選擇分段文字檔",
                "",
                "Text (*.txt);;All files (*)",
            )
            if not transcript:
                return

            try:
                segment_count = len(read_transcript(Path(transcript)))
            except Exception as exc:
                QMessageBox.warning(self, "文字檔錯誤", str(exc))
                return

            output = Path(source) / f"emotion_analysis_result_{_safe_name(name)}.xlsx"
            entry = AnalysisSet(
                name=name.strip(),
                source_directory=Path(source),
                reference_audio=Path(reference),
                transcript_file=Path(transcript),
                segment_count=segment_count,
                output_excel=output,
            )
            self.entries.append(entry)
            self.refresh_table()
            self.status.setText(f"目前共 {len(self.entries)} 個 Analysis Set。")

        def refresh_table(self):
            self.table.setRowCount(len(self.entries))
            for row, entry in enumerate(self.entries):
                values = [
                    entry.name,
                    str(entry.source_directory),
                    str(entry.reference_audio),
                    str(entry.transcript_file),
                    str(entry.segment_count),
                    str(entry.output_excel),
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(value)
                    self.table.setItem(row, column, item)

        def remove_selected(self):
            rows = sorted(
                {index.row() for index in self.table.selectionModel().selectedRows()},
                reverse=True,
            )
            for row in rows:
                del self.entries[row]
            self.refresh_table()
            self.status.setText(f"目前共 {len(self.entries)} 個 Analysis Set。")

        def clear_sets(self):
            if self.process is not None:
                return
            self.entries.clear()
            self.refresh_table()
            self.status.setText("請新增至少一個 Analysis Set。")

        def build_configs(self) -> list[AnalysisConfig]:
            model_name = str(self.model.currentData())
            return [
                AnalysisConfig(
                    reference_audio=entry.reference_audio,
                    segment_count=entry.segment_count,
                    transcript_file=entry.transcript_file,
                    source_directory=entry.source_directory,
                    match_threshold=self.threshold.value(),
                    output_excel=entry.output_excel,
                    emotion_model_name=model_name,
                    segment_padding_seconds=self.padding.value(),
                    noise_reduction_enabled=self.noise_reduction.isChecked(),
                )
                for entry in self.entries
            ]

        def start_job(self, phase: str):
            if self.process is not None:
                return
            if not self.entries:
                QMessageBox.warning(self, "資料不足", "請先新增至少一個 Analysis Set。")
                return
            try:
                configs = self.build_configs()
                self.job_file = create_job_file(configs, phase=phase)
            except Exception as exc:
                QMessageBox.warning(self, "無法建立工作", str(exc))
                return

            self.stdout_buffer = ""
            self.log.clear()
            self.progress.setValue(0)
            self.preprocess_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            self.cancel_button.setEnabled(True)

            process = QProcess(self)
            self.process = process
            process.setProgram(sys.executable)
            process.setArguments(["-m", "emotion_analyzer.worker", str(self.job_file)])
            process.setProcessChannelMode(QProcess.SeparateChannels)
            process.readyReadStandardOutput.connect(self.read_stdout)
            process.readyReadStandardError.connect(self.read_stderr)
            process.finished.connect(self.process_finished)
            self.status.setText(
                "正在切段全部 Analysis Set..."
                if phase == "preprocess"
                else "正在分析全部 Analysis Set..."
            )
            process.start()

        def read_stdout(self):
            if self.process is None:
                return
            text = bytes(self.process.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            )
            self.stdout_buffer += text
            lines = self.stdout_buffer.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                self.stdout_buffer = lines.pop()
            else:
                self.stdout_buffer = ""
            for raw in lines:
                self.handle_line(raw.strip())

        def handle_line(self, line: str):
            event = decode_event(line)
            if event is None:
                if line:
                    self.log.appendPlainText(line)
                return
            kind = str(event.get("event", ""))
            if kind == "progress":
                self.progress.setValue(int(event.get("value", 0)))
                message = str(event.get("message", ""))
                self.status.setText(message)
                self.log.appendPlainText(message)
            elif kind == "failed":
                message = str(event.get("message", "分析失敗"))
                self.status.setText(message)
                self.log.appendPlainText("ERROR: " + message)
            elif kind == "finished":
                self.progress.setValue(100)
                missing = list(event.get("missing_chopped_paths", []) or [])
                errors = list(event.get("errors", []) or [])
                outputs = list(event.get("outputs", []) or [])
                output = event.get("output")
                if output:
                    outputs.append(str(output))
                if missing:
                    self.log.appendPlainText(
                        f"仍缺少 {len(missing)} 個片段，請依下列路徑人工補入："
                    )
                    for path in missing:
                        self.log.appendPlainText(str(path))
                if errors:
                    self.log.appendPlainText("切段警告：")
                    for message in errors:
                        self.log.appendPlainText(str(message))
                if outputs:
                    self.log.appendPlainText("輸出：")
                    for path in outputs:
                        self.log.appendPlainText(str(path))
                self.status.setText(
                    f"完成。缺少 {len(missing)} 個片段。"
                    if missing
                    else "完成。"
                )

        def read_stderr(self):
            if self.process is None:
                return
            text = bytes(self.process.readAllStandardError()).decode(
                "utf-8", errors="replace"
            )
            if text.strip():
                self.log.appendPlainText(text.rstrip())

        def process_finished(self, _exit_code: int, _exit_status):
            if self.stdout_buffer.strip():
                self.handle_line(self.stdout_buffer.strip())
            self.stdout_buffer = ""
            if self.job_file is not None:
                try:
                    self.job_file.unlink(missing_ok=True)
                except OSError:
                    pass
            self.job_file = None
            if self.process is not None:
                self.process.deleteLater()
            self.process = None
            self.preprocess_button.setEnabled(True)
            self.analyze_button.setEnabled(True)
            self.cancel_button.setEnabled(False)

        def cancel_job(self):
            if self.process is None:
                return
            self.status.setText("正在取消...")
            self.process.kill()

        def closeEvent(self, event):
            if self.process is not None:
                self.process.kill()
                self.process.waitForFinished(1500)
            if self.job_file is not None:
                try:
                    self.job_file.unlink(missing_ok=True)
                except OSError:
                    pass
            event.accept()

    app = QApplication.instance() or QApplication(sys.argv)
    window = MultiSetWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_multi_gui())
