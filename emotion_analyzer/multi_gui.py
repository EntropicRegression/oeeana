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


@dataclass
class AnalysisSet:
    name: str
    source_directory: Path | None = None
    reference_audio: Path | None = None
    transcript_file: Path | None = None
    segment_count: int = 0
    output_excel: Path | None = None

    @property
    def is_complete(self) -> bool:
        return (
            self.source_directory is not None
            and self.reference_audio is not None
            and self.transcript_file is not None
            and self.segment_count > 0
            and self.output_excel is not None
        )


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
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QInputDialog,
            QLabel,
            QLineEdit,
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
            self.resize(1320, 860)
            self.entries: list[AnalysisSet] = []
            self.process: QProcess | None = None
            self.job_file: Path | None = None
            self.stdout_buffer = ""
            self.current_set_index: int | None = None

            root = QWidget()
            layout = QVBoxLayout(root)

            hint = QLabel(
                "先新增 Analysis Set，再選取 Set 並於下方設定來源資料夾、標準音檔與分段文字。"
                "每個 Set 可使用不同標準音檔、文字與段落數。"
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)

            self.table = QTableWidget(0, 7)
            self.table.setHorizontalHeaderLabels(
                ["Set", "狀態", "來源資料夾", "標準音檔", "文字檔", "段落數", "輸出"]
            )
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.itemSelectionChanged.connect(self.load_selected_set)
            layout.addWidget(self.table, 2)

            set_buttons = QHBoxLayout()
            add_button = QPushButton("新增 Analysis Set")
            add_button.clicked.connect(self.add_set)
            rename_button = QPushButton("重新命名")
            rename_button.clicked.connect(self.rename_selected)
            remove_button = QPushButton("移除選取")
            remove_button.clicked.connect(self.remove_selected)
            clear_button = QPushButton("全部清除")
            clear_button.clicked.connect(self.clear_sets)
            set_buttons.addWidget(add_button)
            set_buttons.addWidget(rename_button)
            set_buttons.addWidget(remove_button)
            set_buttons.addWidget(clear_button)
            set_buttons.addStretch(1)
            layout.addLayout(set_buttons)

            self.editor_group = QGroupBox("Set 設定")
            editor_form = QFormLayout(self.editor_group)

            self.editor_name = QLabel("尚未選取 Set")
            editor_form.addRow("Set 名稱", self.editor_name)

            self.source_field = QLineEdit()
            self.source_field.setReadOnly(True)
            editor_form.addRow(
                "來源錄音資料夾",
                self._picker_row(self.source_field, "選擇資料夾", self.pick_source_directory),
            )

            self.reference_field = QLineEdit()
            self.reference_field.setReadOnly(True)
            editor_form.addRow(
                "標準音檔",
                self._picker_row(self.reference_field, "選擇音檔", self.pick_reference_audio),
            )

            self.transcript_field = QLineEdit()
            self.transcript_field.setReadOnly(True)
            editor_form.addRow(
                "分段文字檔",
                self._picker_row(self.transcript_field, "選擇 TXT", self.pick_transcript_file),
            )

            self.segment_label = QLabel("-")
            editor_form.addRow("段落數", self.segment_label)

            self.output_field = QLineEdit()
            self.output_field.setReadOnly(True)
            editor_form.addRow("輸出 Excel", self.output_field)

            self.editor_status = QLabel("新增並選取一個 Set 後即可設定。")
            self.editor_status.setWordWrap(True)
            editor_form.addRow("設定狀態", self.editor_status)
            layout.addWidget(self.editor_group)

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
            self.status = QLabel("請先新增 Analysis Set。")
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.progress)
            layout.addWidget(self.status)
            layout.addWidget(self.log, 1)

            self.setCentralWidget(root)
            self.set_editor_enabled(False)

        def _picker_row(self, field: QLineEdit, button_text: str, callback) -> QWidget:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(field, 1)
            button = QPushButton(button_text)
            button.clicked.connect(callback)
            row_layout.addWidget(button)
            return row

        def set_editor_enabled(self, enabled: bool):
            self.editor_group.setEnabled(enabled)

        def add_set(self):
            name, ok = QInputDialog.getText(
                self, "新增 Analysis Set", "Set 名稱，例如 part1 / 問題1："
            )
            name = name.strip()
            if not ok or not name:
                return
            if any(entry.name.casefold() == name.casefold() for entry in self.entries):
                QMessageBox.warning(self, "名稱重複", "Analysis Set 名稱不可重複。")
                return

            self.entries.append(AnalysisSet(name=name))
            row = len(self.entries) - 1
            self.refresh_table(select_row=row)
            self.status.setText(
                f"已新增 {name}。請在 Set 設定區選擇來源資料夾、標準音檔與文字檔。"
            )

        def rename_selected(self):
            entry = self.current_entry()
            if entry is None:
                QMessageBox.information(self, "尚未選取", "請先選取一個 Analysis Set。")
                return
            name, ok = QInputDialog.getText(
                self, "重新命名 Analysis Set", "Set 名稱：", text=entry.name
            )
            name = name.strip()
            if not ok or not name or name == entry.name:
                return
            if any(
                index != self.current_set_index and item.name.casefold() == name.casefold()
                for index, item in enumerate(self.entries)
            ):
                QMessageBox.warning(self, "名稱重複", "Analysis Set 名稱不可重複。")
                return
            entry.name = name
            self.update_output_path(entry)
            row = self.current_set_index
            self.refresh_table(select_row=row)

        def current_entry(self) -> AnalysisSet | None:
            if self.current_set_index is None:
                return None
            if not 0 <= self.current_set_index < len(self.entries):
                return None
            return self.entries[self.current_set_index]

        def load_selected_set(self):
            selected = self.table.selectionModel().selectedRows()
            if len(selected) != 1:
                self.current_set_index = None
                self.clear_editor()
                self.set_editor_enabled(False)
                return
            row = selected[0].row()
            if not 0 <= row < len(self.entries):
                return
            self.current_set_index = row
            self.set_editor_enabled(True)
            self.render_editor(self.entries[row])

        def clear_editor(self):
            self.editor_name.setText("尚未選取 Set")
            self.source_field.clear()
            self.reference_field.clear()
            self.transcript_field.clear()
            self.segment_label.setText("-")
            self.output_field.clear()
            self.editor_status.setText("新增並選取一個 Set 後即可設定。")

        def render_editor(self, entry: AnalysisSet):
            self.editor_name.setText(entry.name)
            self.source_field.setText(str(entry.source_directory) if entry.source_directory else "")
            self.reference_field.setText(str(entry.reference_audio) if entry.reference_audio else "")
            self.transcript_field.setText(str(entry.transcript_file) if entry.transcript_file else "")
            self.segment_label.setText(str(entry.segment_count) if entry.segment_count else "-")
            self.output_field.setText(str(entry.output_excel) if entry.output_excel else "")
            missing = []
            if entry.source_directory is None:
                missing.append("來源資料夾")
            if entry.reference_audio is None:
                missing.append("標準音檔")
            if entry.transcript_file is None:
                missing.append("分段文字檔")
            self.editor_status.setText(
                "設定完成，可執行分析。"
                if not missing
                else "尚需設定：" + "、".join(missing)
            )

        def pick_source_directory(self):
            entry = self.current_entry()
            if entry is None:
                return
            start = str(entry.source_directory or Path.cwd())
            source = QFileDialog.getExistingDirectory(
                self, f"[{entry.name}] 選擇來源錄音資料夾", start
            )
            if not source:
                return
            entry.source_directory = Path(source)
            self.update_output_path(entry)
            self.refresh_current_entry()

        def pick_reference_audio(self):
            entry = self.current_entry()
            if entry is None:
                return
            start = str(entry.reference_audio.parent) if entry.reference_audio else ""
            reference, _ = QFileDialog.getOpenFileName(
                self,
                f"[{entry.name}] 選擇標準音檔",
                start,
                "Audio (*.wav *.mp3 *.m4a *.flac)",
            )
            if not reference:
                return
            entry.reference_audio = Path(reference)
            self.refresh_current_entry()

        def pick_transcript_file(self):
            entry = self.current_entry()
            if entry is None:
                return
            start = str(entry.transcript_file.parent) if entry.transcript_file else ""
            transcript, _ = QFileDialog.getOpenFileName(
                self,
                f"[{entry.name}] 選擇分段文字檔",
                start,
                "Text (*.txt);;All files (*)",
            )
            if not transcript:
                return
            path = Path(transcript)
            try:
                segment_count = len(read_transcript(path))
            except Exception as exc:
                QMessageBox.warning(self, "文字檔錯誤", str(exc))
                return
            entry.transcript_file = path
            entry.segment_count = segment_count
            self.refresh_current_entry()

        def update_output_path(self, entry: AnalysisSet):
            if entry.source_directory is None:
                entry.output_excel = None
                return
            entry.output_excel = (
                entry.source_directory
                / f"emotion_analysis_result_{_safe_name(entry.name)}.xlsx"
            )

        def refresh_current_entry(self):
            row = self.current_set_index
            self.refresh_table(select_row=row)

        def refresh_table(self, select_row: int | None = None):
            self.table.blockSignals(True)
            self.table.setRowCount(len(self.entries))
            for row, entry in enumerate(self.entries):
                values = [
                    entry.name,
                    "完成" if entry.is_complete else "未完成",
                    str(entry.source_directory) if entry.source_directory else "未設定",
                    str(entry.reference_audio) if entry.reference_audio else "未設定",
                    str(entry.transcript_file) if entry.transcript_file else "未設定",
                    str(entry.segment_count) if entry.segment_count else "-",
                    str(entry.output_excel) if entry.output_excel else "未設定",
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(value)
                    self.table.setItem(row, column, item)
            self.table.blockSignals(False)

            if select_row is not None and 0 <= select_row < len(self.entries):
                self.table.selectRow(select_row)
                self.current_set_index = select_row
                self.set_editor_enabled(True)
                self.render_editor(self.entries[select_row])
            elif not self.entries:
                self.current_set_index = None
                self.clear_editor()
                self.set_editor_enabled(False)

        def remove_selected(self):
            row = self.current_set_index
            if row is None:
                return
            del self.entries[row]
            if self.entries:
                next_row = min(row, len(self.entries) - 1)
                self.refresh_table(select_row=next_row)
            else:
                self.refresh_table()
            self.status.setText(f"目前共 {len(self.entries)} 個 Analysis Set。")

        def clear_sets(self):
            if self.process is not None:
                return
            self.entries.clear()
            self.refresh_table()
            self.status.setText("請先新增 Analysis Set。")

        def incomplete_sets(self) -> list[str]:
            return [entry.name for entry in self.entries if not entry.is_complete]

        def build_configs(self) -> list[AnalysisConfig]:
            incomplete = self.incomplete_sets()
            if incomplete:
                raise ValueError(
                    "下列 Analysis Set 尚未完成設定：" + "、".join(incomplete)
                )
            model_name = str(self.model.currentData())
            configs: list[AnalysisConfig] = []
            for entry in self.entries:
                assert entry.reference_audio is not None
                assert entry.transcript_file is not None
                assert entry.source_directory is not None
                assert entry.output_excel is not None
                configs.append(
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
                )
            return configs

        def start_job(self, phase: str):
            if self.process is not None:
                return
            if not self.entries:
                QMessageBox.warning(self, "資料不足", "請先新增至少一個 Analysis Set。")
                return
            incomplete = self.incomplete_sets()
            if incomplete:
                QMessageBox.warning(
                    self,
                    "Set 尚未完成",
                    "請先完成下列 Set 的來源資料夾、標準音檔與文字檔設定：\n"
                    + "\n".join(incomplete),
                )
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
