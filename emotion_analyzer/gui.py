from __future__ import annotations

import sys
from pathlib import Path

from .analytics import (
    DistanceMetric,
    PairingMode,
    compare_runs,
    create_pairing_rows,
    merge_run_analyses,
)
from .core import (
    AnalysisConfig,
    DEFAULT_EMOTION_MODEL,
    EMOTION_MODEL_OPTIONS,
    clear_chopped_directories,
    read_transcript,
)
from .reporting import read_run_report, write_comparison_report
from .worker import create_job_file, decode_event


def run_gui() -> int:
    try:
        from PySide6.QtCharts import (
            QBarCategoryAxis,
            QBarSeries,
            QBarSet,
            QChart,
            QChartView,
            QLineSeries,
            QValueAxis,
        )
        from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
        from PySide6.QtGui import QPainter
        from PySide6.QtWidgets import (
            QApplication,
            QAbstractItemView,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QHeaderView,
            QGroupBox,
            QLabel,
            QLineEdit,
            QListView,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QPlainTextEdit,
            QSpinBox,
            QDoubleSpinBox,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QTreeView,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        print("缺少 PySide6，請先安裝 requirements.txt。", file=sys.stderr)
        return 1

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("emotion2vec+ 批次情緒分析")
            self.resize(1180, 780)
            self.process: QProcess | None = None
            self.job_file: Path | None = None
            self.process_stdout_buffer = ""
            self.process_error_tail: list[str] = []
            self.process_failure_message: str | None = None
            self.analysis_outputs: list[str] = []
            self.cancel_requested = False
            self.comparison_first_run = None
            self.comparison_second_run = None
            self.current_comparison = None

            self.reference = QLineEdit()
            self.transcript = QLineEdit()
            self.source_directories = QListWidget()
            self.source_directories.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.source_directories.setMinimumHeight(105)
            self.source_directories.setToolTip("可用 Ctrl 或 Shift 一次選取多個同次實驗資料夾")
            self.segment_count = QSpinBox()
            self.segment_count.setRange(1, 10000)
            self.segment_count.setValue(1)
            self.segment_count.setReadOnly(True)
            self.segment_count.setToolTip("選擇分段文字檔後自動計算")
            self.threshold = QDoubleSpinBox()
            self.threshold.setRange(0.0, 1.0)
            self.threshold.setSingleStep(0.05)
            self.threshold.setDecimals(2)
            self.threshold.setValue(0.30)
            self.threshold.setToolTip("主要匹配門檻；全篇順序與前後文成立時，最多自動放寬 0.05")
            self.segment_padding = QDoubleSpinBox()
            self.segment_padding.setRange(0.0, 5.0)
            self.segment_padding.setSingleStep(0.1)
            self.segment_padding.setDecimals(1)
            self.segment_padding.setValue(1.0)
            self.segment_padding.setSuffix(" 秒")
            self.segment_padding.setToolTip("每段開始前與結束後各延伸此時間；相鄰片段可以重疊")
            self.emotion_model = QComboBox()
            for label, model_name in EMOTION_MODEL_OPTIONS:
                self.emotion_model.addItem(label, model_name)
            self.emotion_model.setCurrentIndex(
                max(0, self.emotion_model.findData(DEFAULT_EMOTION_MODEL))
            )
            self.emotion_model.setToolTip(
                "large 為目前預設；未安裝於 models/ 的版本會由 ModelScope 載入"
            )
            self.noise_reduction = QCheckBox("啟用保守型語音去雜音")
            self.noise_reduction.setChecked(True)
            self.noise_reduction.setToolTip(
                "在 Whisper 分段與情緒分析前，先降低持續性的風扇、冷氣與底噪"
            )
            self.progress_bar = QProgressBar()
            self.status = QLabel("就緒")
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.start_button = QPushButton("開始分析")
            self.cancel_button = QPushButton("取消")
            self.cancel_button.setEnabled(False)
            self.reset_button = QPushButton("重置介面")
            self.reset_button.setToolTip("只清空介面，不會刪除音檔、切段快取或 Excel")

            form = QFormLayout()
            form.addRow("標準音檔", self._picker_row(self.reference, self.pick_reference, "選擇音檔"))
            form.addRow("分段文字檔", self._picker_row(self.transcript, self.pick_transcript, "選擇 TXT"))
            source_panel = QWidget()
            source_layout = QVBoxLayout(source_panel)
            source_layout.setContentsMargins(0, 0, 0, 0)
            source_layout.addWidget(self.source_directories)
            source_buttons = QHBoxLayout()
            self.source_add_button = QPushButton("加入資料夾（可複選）")
            self.source_add_button.clicked.connect(self.pick_source)
            self.source_remove_button = QPushButton("移除選取")
            self.source_remove_button.clicked.connect(self.remove_selected_sources)
            self.source_clear_button = QPushButton("清除清單內全部 chopped")
            self.source_clear_button.clicked.connect(self.clear_selected_chopped)
            source_buttons.addWidget(self.source_add_button)
            source_buttons.addWidget(self.source_remove_button)
            source_buttons.addWidget(self.source_clear_button)
            source_layout.addLayout(source_buttons)
            form.addRow("同次實驗資料夾", source_panel)
            form.addRow("分段數量", self.segment_count)
            form.addRow("Whisper 文字匹配門檻", self.threshold)
            form.addRow("切段前後緩衝（可重疊）", self.segment_padding)
            form.addRow("emotion2vec+ 模型", self.emotion_model)
            form.addRow("背景雜音處理", self.noise_reduction)

            buttons = QHBoxLayout()
            buttons.addWidget(self.start_button)
            buttons.addWidget(self.cancel_button)
            buttons.addWidget(self.reset_button)
            self.start_button.clicked.connect(self.start_analysis)
            self.cancel_button.clicked.connect(self.cancel_analysis)
            self.reset_button.clicked.connect(self.reset_interface)
            self.transcript.editingFinished.connect(self.refresh_transcript_segment_count)

            audio_tab = QWidget()
            audio_layout = QVBoxLayout(audio_tab)
            audio_layout.addLayout(form)
            audio_layout.addLayout(buttons)
            audio_layout.addWidget(self.progress_bar)
            audio_layout.addWidget(self.status)
            audio_layout.addWidget(self.log)

            tabs = QTabWidget()
            tabs.addTab(audio_tab, "音檔分析")
            tabs.addTab(self._build_comparison_tab(), "前後測資料與分組")
            tabs.addTab(self._build_interactive_tab(), "互動分析")
            self.main_tabs = tabs

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.addWidget(tabs)
            self.setCentralWidget(central)

        def _picker_row(self, field: QLineEdit, callback, label: str) -> QWidget:
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            button = QPushButton(label)
            button.clicked.connect(callback)
            layout.addWidget(field, 1)
            layout.addWidget(button)
            return row

        def _build_comparison_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            source_row = QHBoxLayout()
            self.compare_pre_files = QListWidget()
            self.compare_post_files = QListWidget()
            for widget in (self.compare_pre_files, self.compare_post_files):
                widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
                widget.setMinimumHeight(105)

            pre_group = QGroupBox("前測 Excel（可匯入多份）")
            pre_layout = QVBoxLayout(pre_group)
            pre_layout.addWidget(self.compare_pre_files)
            pre_buttons = QHBoxLayout()
            add_pre = QPushButton("加入檔案")
            add_pre.clicked.connect(self.pick_compare_pre)
            remove_pre = QPushButton("移除選取")
            remove_pre.clicked.connect(lambda: self.remove_report_files(self.compare_pre_files))
            pre_buttons.addWidget(add_pre)
            pre_buttons.addWidget(remove_pre)
            pre_layout.addLayout(pre_buttons)

            post_group = QGroupBox("後測 Excel（可匯入多份）")
            post_layout = QVBoxLayout(post_group)
            post_layout.addWidget(self.compare_post_files)
            post_buttons = QHBoxLayout()
            add_post = QPushButton("加入檔案")
            add_post.clicked.connect(self.pick_compare_post)
            remove_post = QPushButton("移除選取")
            remove_post.clicked.connect(lambda: self.remove_report_files(self.compare_post_files))
            post_buttons.addWidget(add_post)
            post_buttons.addWidget(remove_post)
            post_layout.addLayout(post_buttons)
            source_row.addWidget(pre_group)
            source_row.addWidget(post_group)
            layout.addLayout(source_row)

            self.compare_mode = QComboBox()
            self.compare_mode.addItem("依受試者名稱", PairingMode.NAME.value)
            self.compare_mode.addItem("依資料順序", PairingMode.ORDER.value)
            self.compare_mode.addItem("手動一對一配對", PairingMode.MANUAL.value)
            self.compare_mode.currentIndexChanged.connect(self._comparison_mode_changed)
            self.compare_metric = QComboBox()
            self.compare_metric.addItem("L1 情緒截距（Friendly Support）", DistanceMetric.L1.value)
            self.compare_metric.addItem("L2 情緒差距（oeeana 原格式）", DistanceMetric.L2.value)
            self.compare_metric.currentIndexChanged.connect(self._comparison_metric_changed)

            form = QFormLayout()
            form.addRow("配對方式", self.compare_mode)
            form.addRow("比較指標", self.compare_metric)
            layout.addLayout(form)

            button_row = QHBoxLayout()
            load_button = QPushButton("載入並建立配對")
            load_button.clicked.connect(self.load_comparison_pairings)
            self.compare_export_button = QPushButton("輸出比較報告")
            self.compare_export_button.setEnabled(False)
            self.compare_export_button.clicked.connect(self.export_comparison_report)
            self.compare_refresh_button = QPushButton("更新互動分析")
            self.compare_refresh_button.setEnabled(False)
            self.compare_refresh_button.clicked.connect(self.refresh_interactive_analysis)
            button_row.addWidget(load_button)
            button_row.addWidget(self.compare_refresh_button)
            button_row.addWidget(self.compare_export_button)
            layout.addLayout(button_row)

            group_buttons = QHBoxLayout()
            group_buttons.addWidget(QLabel("選取多列後批次分組："))
            for label, group in (
                ("設為實驗組", "實驗組"),
                ("設為對照組", "對照組"),
                ("設為未分組", "未分組"),
                ("排除", "排除"),
            ):
                button = QPushButton(label)
                button.clicked.connect(lambda _checked=False, value=group: self.set_selected_group(value))
                group_buttons.addWidget(button)
            group_buttons.addStretch(1)
            layout.addLayout(group_buttons)

            self.compare_table = QTableWidget(0, 6)
            self.compare_table.setHorizontalHeaderLabels(
                ["前測受試者", "前測來源", "後測受試者", "後測來源", "群組", "資料狀態"]
            )
            self.compare_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.compare_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.compare_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.compare_table)

            self.compare_status = QLabel("請分別加入一份以上的前測與後測報表。")
            self.compare_result = QPlainTextEdit()
            self.compare_result.setReadOnly(True)
            self.compare_result.setMaximumHeight(220)
            layout.addWidget(self.compare_status)
            layout.addWidget(self.compare_result)
            self.loaded_pairing_mode = None
            return tab

        def _build_interactive_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            controls = QHBoxLayout()
            controls.addWidget(QLabel("分析範圍"))
            self.analysis_scope = QComboBox()
            self.analysis_scope.addItem("整體（排除已標記排除者）", "整體")
            self.analysis_scope.addItem("實驗組", "實驗組")
            self.analysis_scope.addItem("對照組", "對照組")
            self.analysis_scope.currentIndexChanged.connect(self.render_interactive_analysis)
            controls.addWidget(self.analysis_scope)
            controls.addWidget(QLabel("分段趨勢受試者"))
            self.analysis_subject = QComboBox()
            self.analysis_subject.currentIndexChanged.connect(self.render_segment_chart)
            controls.addWidget(self.analysis_subject, 1)
            layout.addLayout(controls)

            self.analysis_summary = QPlainTextEdit()
            self.analysis_summary.setReadOnly(True)
            self.analysis_summary.setMaximumHeight(105)
            self.analysis_summary.setPlainText("請先在「前後測資料與分組」載入報表。")
            layout.addWidget(self.analysis_summary)

            group_test_box = QGroupBox("組間效果檢定")
            group_test_layout = QVBoxLayout(group_test_box)
            self.group_effect_summary = QPlainTextEdit()
            self.group_effect_summary.setReadOnly(True)
            self.group_effect_summary.setMaximumHeight(145)
            self.group_effect_summary.setPlainText("完成實驗組與對照組分組後顯示 ANCOVA 與 Welch t 檢定。")
            group_test_layout.addWidget(self.group_effect_summary)
            layout.addWidget(group_test_box)

            self.analysis_chart_tabs = QTabWidget()
            self.group_chart_view = self._chart_view("尚未建立整體與分組分析")
            self.subject_chart_view = self._chart_view("尚未建立受試者比較")
            self.segment_chart_view = self._chart_view("尚未選擇受試者")
            self.analysis_chart_tabs.addTab(self.group_chart_view, "整體與分組")
            self.analysis_chart_tabs.addTab(self.subject_chart_view, "受試者前後測")
            self.analysis_chart_tabs.addTab(self.segment_chart_view, "分段趨勢")
            layout.addWidget(self.analysis_chart_tabs, 2)

            self.analysis_table = QTableWidget(0, 8)
            self.analysis_table.setHorizontalHeaderLabels(
                ["受試者", "群組", "前測來源", "前測", "後測來源", "後測", "後測－前測", "成對"]
            )
            self.analysis_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.analysis_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.analysis_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(self.analysis_table, 1)
            return tab

        def _chart_view(self, message: str):
            chart = QChart()
            chart.setTitle(message)
            view = QChartView(chart)
            view.setRenderHint(QPainter.Antialiasing)
            view.setRubberBand(QChartView.RectangleRubberBand)
            return view

        def _comparison_mode_changed(self, *_args):
            if hasattr(self, "compare_export_button"):
                self.compare_export_button.setEnabled(False)
                self.compare_refresh_button.setEnabled(False)
            if hasattr(self, "compare_status"):
                self.compare_status.setText("配對方式已變更，請重新載入配對。")
            self.loaded_pairing_mode = None
            self.current_comparison = None

        def _comparison_metric_changed(self, *_args):
            if (
                not hasattr(self, "compare_table")
                or self.comparison_first_run is None
                or self.comparison_second_run is None
            ):
                return
            metric = DistanceMetric(str(self.compare_metric.currentData()))
            common_metrics = (
                self.comparison_first_run.available_metrics
                & self.comparison_second_run.available_metrics
            )
            if metric not in common_metrics:
                self.current_comparison = None
                self.compare_status.setText(
                    f"前測與後測沒有共同可用的 {metric.display_name}，請改選其他指標。"
                )
                self._reset_interactive_analysis(
                    f"目前匯入的資料無法使用 {metric.display_name}。"
                )
                return
            mode = PairingMode(str(self.compare_mode.currentData()))
            for row in range(self.compare_table.rowCount()):
                first_item = self.compare_table.item(row, 0)
                first_key = (
                    str(first_item.data(Qt.ItemDataRole.UserRole) or "")
                    if first_item
                    else ""
                )
                first_subject = self._subject_by_record(
                    self.comparison_first_run, first_key
                )
                if mode is PairingMode.MANUAL:
                    second_combo = self.compare_table.cellWidget(row, 2)
                    second_key = (
                        str(second_combo.currentData() or "")
                        if isinstance(second_combo, QComboBox)
                        else ""
                    )
                else:
                    second_item = self.compare_table.item(row, 2)
                    second_key = (
                        str(second_item.data(Qt.ItemDataRole.UserRole) or "")
                        if second_item
                        else ""
                    )
                second_subject = self._subject_by_record(
                    self.comparison_second_run, second_key
                )
                self._set_row_data_status(row, first_subject, second_subject, metric)
            if self.loaded_pairing_mode is not None:
                self.refresh_interactive_analysis(show_error=False)

        def _group_combo(self, current: str = "未分組") -> QComboBox:
            combo = QComboBox()
            for value in ("未分組", "實驗組", "對照組", "排除"):
                combo.addItem(value, value)
            index = combo.findData(current)
            combo.setCurrentIndex(max(index, 0))
            return combo

        def _report_paths(self, widget: QListWidget) -> list[Path]:
            paths: list[Path] = []
            for index in range(widget.count()):
                item = widget.item(index)
                path = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
                if path:
                    paths.append(Path(path))
            return paths

        def _add_report_files(self, widget: QListWidget, title: str):
            paths, _ = QFileDialog.getOpenFileNames(self, title, "", "Excel (*.xlsx)")
            if not paths:
                return
            existing = {str(path.resolve()).casefold() for path in self._report_paths(widget)}
            for raw_path in paths:
                path = Path(raw_path).resolve()
                if str(path).casefold() in existing:
                    continue
                item = QListWidgetItem(path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                item.setToolTip(str(path))
                widget.addItem(item)
                existing.add(str(path).casefold())
            self._invalidate_comparison("報表清單已變更，請重新載入配對。")

        def pick_compare_pre(self):
            self._add_report_files(self.compare_pre_files, "選擇前測分析報表（可複選）")

        def pick_compare_post(self):
            self._add_report_files(self.compare_post_files, "選擇後測分析報表（可複選）")

        def remove_report_files(self, widget: QListWidget):
            selected = widget.selectedItems()
            if not selected:
                return
            for item in selected:
                widget.takeItem(widget.row(item))
            self._invalidate_comparison("報表清單已變更，請重新載入配對。")

        def _invalidate_comparison(self, message: str):
            self.comparison_first_run = None
            self.comparison_second_run = None
            self.current_comparison = None
            self.loaded_pairing_mode = None
            self.compare_table.setRowCount(0)
            self.compare_result.clear()
            self.compare_refresh_button.setEnabled(False)
            self.compare_export_button.setEnabled(False)
            self.compare_status.setText(message)
            self._reset_interactive_analysis()

        @staticmethod
        def _subject_source(subject) -> str:
            return subject.source_name or ""

        @staticmethod
        def _set_subject_item(table: QTableWidget, row: int, column: int, subject):
            item = QTableWidgetItem(subject.subject_id if subject else "")
            item.setData(Qt.ItemDataRole.UserRole, subject.record_key if subject else "")
            table.setItem(row, column, item)

        def _update_manual_post_source(self, row: int, combo: QComboBox):
            record_key = str(combo.currentData() or "")
            source = ""
            if self.comparison_second_run is not None:
                subject = next(
                    (
                        candidate
                        for candidate in self.comparison_second_run.comparison_subjects
                        if candidate.record_key == record_key
                    ),
                    None,
                )
                source = self._subject_source(subject) if subject else ""
            self.compare_table.setItem(row, 3, QTableWidgetItem(source))

        @staticmethod
        def _subject_by_record(run, record_key: str):
            if run is None or not record_key:
                return None
            return next(
                (
                    subject
                    for subject in run.comparison_subjects
                    if subject.record_key == record_key
                ),
                None,
            )

        @staticmethod
        def _subject_data_issues(subject, run, metric, phase: str) -> list[str]:
            if subject is None:
                return [f"缺少{phase}配對"]
            issues: list[str] = []
            if subject.error:
                issues.append(f"{phase}分析錯誤：{subject.error}")
            if subject.metric_value(metric) is None:
                issues.append(f"{phase}缺少{metric.display_name}")
            expected_count = subject.source_segment_count or run.segment_count
            segment_values = {
                segment.segment_index: (
                    segment.l1_distance
                    if metric is DistanceMetric.L1
                    else segment.l2_distance
                )
                for segment in subject.segments
                if not segment.error
            }
            missing_segments = [
                str(index)
                for index in range(1, expected_count + 1)
                if segment_values.get(index) is None
            ]
            if missing_segments:
                issues.append(f"{phase}缺少段落 {', '.join(missing_segments)}")
            if any(segment.error for segment in subject.segments):
                issues.append(f"{phase}含分析失敗段落")
            return issues

        def _set_row_data_status(self, row: int, first_subject, second_subject, metric):
            previous_item = self.compare_table.item(row, 5)
            previous_group_role = int(Qt.ItemDataRole.UserRole) + 1
            was_auto_excluded = bool(
                previous_item
                and previous_item.data(Qt.ItemDataRole.UserRole)
            )
            group_combo = self.compare_table.cellWidget(row, 4)
            previous_group = (
                str(previous_item.data(previous_group_role) or "未分組")
                if was_auto_excluded
                else (
                    str(group_combo.currentData() or "未分組")
                    if isinstance(group_combo, QComboBox)
                    else "未分組"
                )
            )
            issues = [
                *self._subject_data_issues(
                    first_subject, self.comparison_first_run, metric, "前測"
                ),
                *self._subject_data_issues(
                    second_subject, self.comparison_second_run, metric, "後測"
                ),
            ]
            reason = "；".join(issues)
            status_item = QTableWidgetItem(reason or "完整")
            status_item.setData(Qt.ItemDataRole.UserRole, reason)
            status_item.setData(previous_group_role, previous_group if reason else "")
            status_item.setToolTip(reason or "資料完整，可納入統計。")
            self.compare_table.setItem(row, 5, status_item)
            if isinstance(group_combo, QComboBox):
                if reason:
                    excluded_index = group_combo.findData("排除")
                    group_combo.setCurrentIndex(excluded_index)
                    group_combo.setEnabled(False)
                    group_combo.setToolTip(f"資料缺損，已自動排除：{reason}")
                else:
                    group_combo.setEnabled(True)
                    group_combo.setToolTip("")
                    if was_auto_excluded and group_combo.currentData() == "排除":
                        restore_index = group_combo.findData(previous_group)
                        group_combo.setCurrentIndex(max(restore_index, 0))

        def _manual_post_changed(self, row: int, combo: QComboBox):
            self._update_manual_post_source(row, combo)
            first_item = self.compare_table.item(row, 0)
            first_key = (
                str(first_item.data(Qt.ItemDataRole.UserRole) or "")
                if first_item
                else ""
            )
            first_subject = self._subject_by_record(self.comparison_first_run, first_key)
            second_subject = self._subject_by_record(
                self.comparison_second_run, str(combo.currentData() or "")
            )
            metric = DistanceMetric(str(self.compare_metric.currentData()))
            self._set_row_data_status(row, first_subject, second_subject, metric)
            if self.loaded_pairing_mode is not None:
                self.refresh_interactive_analysis(show_error=False)

        def load_comparison_pairings(self):
            first_paths = self._report_paths(self.compare_pre_files)
            second_paths = self._report_paths(self.compare_post_files)
            if not first_paths or not second_paths:
                QMessageBox.warning(self, "報表不足", "前測與後測都至少要加入一份 Excel 報表。")
                return
            try:
                first_run = merge_run_analyses(
                    [read_run_report(path) for path in first_paths],
                    [path.name for path in first_paths],
                )
                second_run = merge_run_analyses(
                    [read_run_report(path) for path in second_paths],
                    [path.name for path in second_paths],
                )
                self.comparison_first_run = first_run
                self.comparison_second_run = second_run
                common_metrics = first_run.available_metrics & second_run.available_metrics
                if not common_metrics:
                    raise ValueError("前測與後測沒有共同的距離指標，無法進行比較。")
                selected_metric = DistanceMetric(str(self.compare_metric.currentData()))
                if selected_metric not in common_metrics:
                    selected_metric = (
                        DistanceMetric.L1
                        if DistanceMetric.L1 in common_metrics
                        else DistanceMetric.L2
                    )
                    self.compare_metric.setCurrentIndex(
                        self.compare_metric.findData(selected_metric.value)
                    )
                mode = PairingMode(str(self.compare_mode.currentData()))
                self.compare_table.setRowCount(0)
                if mode is PairingMode.MANUAL:
                    second_subjects = second_run.comparison_subjects
                    used_defaults: set[str] = set()
                    for first_subject in first_run.comparison_subjects:
                        row = self.compare_table.rowCount()
                        self.compare_table.insertRow(row)
                        self._set_subject_item(self.compare_table, row, 0, first_subject)
                        self.compare_table.setItem(
                            row, 1, QTableWidgetItem(self._subject_source(first_subject))
                        )
                        second_combo = QComboBox()
                        second_combo.addItem("（不配對）", "")
                        default_index = 0
                        for index, second_subject in enumerate(second_subjects, 1):
                            label = f"{second_subject.subject_id} — {self._subject_source(second_subject)}"
                            second_combo.addItem(label, second_subject.record_key)
                            if (
                                not default_index
                                and second_subject.subject_id == first_subject.subject_id
                                and second_subject.record_key not in used_defaults
                            ):
                                default_index = index
                                used_defaults.add(second_subject.record_key)
                        second_combo.setCurrentIndex(default_index)
                        second_combo.currentIndexChanged.connect(
                            lambda _index, target_row=row, widget=second_combo:
                            self._manual_post_changed(target_row, widget)
                        )
                        self.compare_table.setCellWidget(row, 2, second_combo)
                        self._update_manual_post_source(row, second_combo)
                        self.compare_table.setCellWidget(row, 4, self._group_combo())
                        second_subject = self._subject_by_record(
                            second_run, str(second_combo.currentData() or "")
                        )
                        self._set_row_data_status(
                            row, first_subject, second_subject, selected_metric
                        )
                else:
                    for first_subject, second_subject in create_pairing_rows(first_run, second_run, mode):
                        row = self.compare_table.rowCount()
                        self.compare_table.insertRow(row)
                        self._set_subject_item(self.compare_table, row, 0, first_subject)
                        self.compare_table.setItem(
                            row, 1, QTableWidgetItem(self._subject_source(first_subject) if first_subject else "")
                        )
                        self._set_subject_item(self.compare_table, row, 2, second_subject)
                        self.compare_table.setItem(
                            row, 3, QTableWidgetItem(self._subject_source(second_subject) if second_subject else "")
                        )
                        self.compare_table.setCellWidget(row, 4, self._group_combo())
                        self._set_row_data_status(
                            row, first_subject, second_subject, selected_metric
                        )
                self.comparison_first_run = first_run
                self.comparison_second_run = second_run
                self.loaded_pairing_mode = mode
                self.current_comparison = None
                first_metrics = ", ".join(
                    metric.value.upper() for metric in sorted(first_run.available_metrics, key=lambda item: item.value)
                ) or "無"
                second_metrics = ", ".join(
                    metric.value.upper() for metric in sorted(second_run.available_metrics, key=lambda item: item.value)
                ) or "無"
                self.compare_status.setText(
                    f"已合併前測 {len(first_paths)} 份、後測 {len(second_paths)} 份，"
                    f"建立 {self.compare_table.rowCount()} 列；可用指標：前測 {first_metrics}，後測 {second_metrics}。"
                )
                self.compare_export_button.setEnabled(True)
                self.compare_refresh_button.setEnabled(True)
                self.refresh_interactive_analysis(show_error=False)
            except Exception as exc:
                self._invalidate_comparison("載入報表失敗。")
                QMessageBox.critical(self, "載入失敗", str(exc))

        def set_selected_group(self, group: str):
            selected_rows = sorted(
                {index.row() for index in self.compare_table.selectionModel().selectedRows()}
            )
            if not selected_rows:
                QMessageBox.warning(self, "尚未選取", "請先在配對表中選取一列或多列資料。")
                return
            updated_count = 0
            skipped_count = 0
            for row in selected_rows:
                combo = self.compare_table.cellWidget(row, 4)
                if isinstance(combo, QComboBox):
                    if not combo.isEnabled() and group != "排除":
                        skipped_count += 1
                        continue
                    index = combo.findData(group)
                    if index >= 0:
                        combo.setCurrentIndex(index)
                        updated_count += 1
            message = f"已將 {updated_count} 列設為「{group}」。"
            if skipped_count:
                message += f"另有 {skipped_count} 列因資料缺損維持排除。"
            self.refresh_interactive_analysis(show_error=False)
            self.compare_status.setText(message)

        def _comparison_inputs(self):
            mode = PairingMode(str(self.compare_mode.currentData()))
            if mode != self.loaded_pairing_mode:
                raise ValueError("配對方式已變更，請重新載入配對。")
            manual_pairs: list[tuple[str, str]] = []
            groups: dict[str, str] = {}
            exclusion_reasons: dict[str, str] = {}
            for row in range(self.compare_table.rowCount()):
                first_item = self.compare_table.item(row, 0)
                first_id = str(first_item.data(Qt.ItemDataRole.UserRole) or "") if first_item else ""
                if mode is PairingMode.MANUAL:
                    second_widget = self.compare_table.cellWidget(row, 2)
                    second_id = str(second_widget.currentData() or "") if second_widget else ""
                    if first_id and second_id:
                        manual_pairs.append((first_id, second_id))
                else:
                    second_item = self.compare_table.item(row, 2)
                    second_id = (
                        str(second_item.data(Qt.ItemDataRole.UserRole) or "") if second_item else ""
                    )
                group_widget = self.compare_table.cellWidget(row, 4)
                group = str(group_widget.currentData() or "未分組") if group_widget else "未分組"
                pairing_key = first_id or second_id
                if pairing_key:
                    groups[pairing_key] = group
                    status_item = self.compare_table.item(row, 5)
                    reason = (
                        str(status_item.data(Qt.ItemDataRole.UserRole) or "")
                        if status_item
                        else ""
                    )
                    if reason:
                        exclusion_reasons[pairing_key] = reason
            if mode is PairingMode.MANUAL and self.comparison_second_run is not None:
                paired_second_ids = {second_id for _first_id, second_id in manual_pairs}
                for subject in self.comparison_second_run.comparison_subjects:
                    if subject.record_key in paired_second_ids:
                        continue
                    groups[subject.record_key] = "排除"
                    exclusion_reasons[subject.record_key] = "缺少前測配對"
            return mode, manual_pairs, groups, exclusion_reasons

        def _build_current_comparison(self):
            if self.comparison_first_run is None or self.comparison_second_run is None:
                raise ValueError("請先載入前測與後測報表並建立配對。")
            mode, manual_pairs, groups, exclusion_reasons = self._comparison_inputs()
            metric = DistanceMetric(str(self.compare_metric.currentData()))
            return compare_runs(
                self.comparison_first_run,
                self.comparison_second_run,
                metric=metric,
                pairing_mode=mode,
                manual_pairs=manual_pairs,
                groups=groups,
                exclusion_reasons=exclusion_reasons,
                first_source=f"前測（{len(self.comparison_first_run.source_files)} 份）",
                second_source=f"後測（{len(self.comparison_second_run.source_files)} 份）",
            )

        @staticmethod
        def _number_text(value) -> str:
            return "N/A" if value is None else f"{value:.4f}"

        def _comparison_summary_text(self, statistics) -> str:
            test = statistics.paired_test
            return (
                f"{statistics.name}｜前測 n={statistics.first.count}、平均 "
                f"{self._number_text(statistics.first.mean)}、SD {self._number_text(statistics.first.standard_deviation)}\n"
                f"後測 n={statistics.second.count}、平均 {self._number_text(statistics.second.mean)}、"
                f"SD {self._number_text(statistics.second.standard_deviation)}\n"
                f"有效成對 n={test.paired_count}、t={self._number_text(test.t_statistic)}、"
                f"p={self._number_text(test.p_value)}｜{test.note}"
            )

        def _group_effect_summary_text(self, comparison) -> str:
            lines: list[str] = []
            for test in comparison.group_effect_tests:
                confidence_interval = (
                    "N/A"
                    if test.confidence_interval_low is None
                    or test.confidence_interval_high is None
                    else (
                        f"[{self._number_text(test.confidence_interval_low)}, "
                        f"{self._number_text(test.confidence_interval_high)}]"
                    )
                )
                lines.append(
                    f"{test.name}｜實驗組 n={test.experimental_count}、平均變化 "
                    f"{self._number_text(test.experimental_mean_change)}；對照組 n={test.control_count}、"
                    f"平均變化 {self._number_text(test.control_mean_change)}"
                )
                lines.append(
                    f"{test.effect_definition}={self._number_text(test.effect)}、"
                    f"SE={self._number_text(test.standard_error)}、t={self._number_text(test.t_statistic)}、"
                    f"df={self._number_text(test.degrees_freedom)}、p={self._number_text(test.p_value)}、"
                    f"95% CI={confidence_interval}｜{test.note}"
                )
            return "\n".join(lines)

        def _comparison_full_summary_text(self, comparison) -> str:
            return (
                f"{self._comparison_summary_text(comparison.overall)}\n\n"
                f"組間效果檢定\n{self._group_effect_summary_text(comparison)}"
            )

        def refresh_interactive_analysis(self, _checked=False, *, show_error: bool = True):
            try:
                comparison = self._build_current_comparison()
                self.current_comparison = comparison
                self.compare_result.setPlainText(self._comparison_full_summary_text(comparison))
                self.render_interactive_analysis()
                excluded_count = sum(
                    1
                    for row in range(self.compare_table.rowCount())
                    if (
                        (item := self.compare_table.item(row, 5)) is not None
                        and item.data(Qt.ItemDataRole.UserRole)
                    )
                )
                suffix = (
                    f"；資料缺損自動排除 {excluded_count} 列"
                    if excluded_count
                    else ""
                )
                self.compare_status.setText(
                    f"互動分析已更新{suffix}；可切換到「互動分析」分頁查看。"
                )
                return comparison
            except Exception as exc:
                self.current_comparison = None
                self._reset_interactive_analysis(str(exc))
                if show_error:
                    QMessageBox.critical(self, "比較失敗", str(exc))
                return None

        def export_comparison_report(self):
            if self.comparison_first_run is None or self.comparison_second_run is None:
                QMessageBox.warning(self, "尚未載入", "請先載入前測與後測報表並建立配對。")
                return
            try:
                comparison = self._build_current_comparison()
                self.current_comparison = comparison
                first_paths = self._report_paths(self.compare_pre_files)
                default_output = first_paths[0].parent / "experiment_comparison.xlsx"
                output, _ = QFileDialog.getSaveFileName(
                    self,
                    "儲存前後測比較報告",
                    str(default_output),
                    "Excel (*.xlsx)",
                )
                if not output:
                    return
                output_path = Path(output)
                if output_path.suffix.casefold() != ".xlsx":
                    output_path = output_path.with_suffix(".xlsx")
                write_comparison_report(output_path, comparison)
                self.compare_result.setPlainText(self._comparison_full_summary_text(comparison))
                self.render_interactive_analysis()
                self.compare_status.setText(f"比較報告已輸出：{output_path}")
                QMessageBox.information(self, "完成", f"前後測比較報告已輸出至：\n{output_path}")
            except Exception as exc:
                QMessageBox.critical(self, "比較失敗", str(exc))

        def _reset_interactive_analysis(self, message: str | None = None):
            text = message or "請先在「前後測資料與分組」載入報表。"
            if not hasattr(self, "analysis_summary"):
                return
            self.analysis_summary.setPlainText(text)
            self.group_effect_summary.setPlainText(
                "完成實驗組與對照組分組後顯示 ANCOVA 與 Welch t 檢定。"
            )
            self.analysis_table.setRowCount(0)
            self.analysis_subject.blockSignals(True)
            self.analysis_subject.clear()
            self.analysis_subject.blockSignals(False)
            for view, title in (
                (self.group_chart_view, "尚未建立整體與分組分析"),
                (self.subject_chart_view, "尚未建立受試者比較"),
                (self.segment_chart_view, "尚未選擇受試者"),
            ):
                chart = QChart()
                chart.setTitle(title)
                view.setChart(chart)

        def _comparison_rows_for_scope(self):
            if self.current_comparison is None:
                return ()
            scope = str(self.analysis_scope.currentData() or "整體")
            if scope == "整體":
                return tuple(row for row in self.current_comparison.rows if row.group != "排除")
            return tuple(row for row in self.current_comparison.rows if row.group == scope)

        def render_interactive_analysis(self, *_args):
            comparison = self.current_comparison
            if comparison is None:
                self._reset_interactive_analysis()
                return
            scope = str(self.analysis_scope.currentData() or "整體")
            if scope == "整體":
                statistics = comparison.overall
            else:
                statistics = next(item for item in comparison.groups if item.name == scope)
            rows = self._comparison_rows_for_scope()
            self.analysis_summary.setPlainText(
                f"指標：{comparison.metric.display_name}\n{self._comparison_summary_text(statistics)}"
            )
            self.group_effect_summary.setPlainText(self._group_effect_summary_text(comparison))

            self.analysis_table.setRowCount(len(rows))
            for table_row, row in enumerate(rows):
                difference = (
                    row.second_value - row.first_value
                    if row.first_value is not None and row.second_value is not None
                    else None
                )
                values = (
                    row.display_name,
                    row.group,
                    row.first_report or "",
                    "" if row.first_value is None else f"{row.first_value:.6f}",
                    row.second_report or "",
                    "" if row.second_value is None else f"{row.second_value:.6f}",
                    "" if difference is None else f"{difference:.6f}",
                    "是" if difference is not None else "否",
                )
                for column, value in enumerate(values):
                    self.analysis_table.setItem(table_row, column, QTableWidgetItem(value))

            self._render_group_chart(comparison)
            self._render_subject_chart(rows, comparison.metric.display_name)

            previous_key = str(self.analysis_subject.currentData() or "")
            self.analysis_subject.blockSignals(True)
            self.analysis_subject.clear()
            for row in rows:
                self.analysis_subject.addItem(
                    f"{row.display_name}｜{row.first_report or row.second_report or ''}",
                    row.pairing_key,
                )
            previous_index = self.analysis_subject.findData(previous_key)
            if previous_index >= 0:
                self.analysis_subject.setCurrentIndex(previous_index)
            self.analysis_subject.blockSignals(False)
            self.render_segment_chart()

        def _render_group_chart(self, comparison):
            statistics_rows = [comparison.overall, *comparison.groups]
            plottable = [
                item
                for item in statistics_rows
                if item.first.mean is not None and item.second.mean is not None
            ]
            chart = QChart()
            chart.setTitle(f"{comparison.metric.display_name}：整體與分組平均")
            if not plottable:
                chart.setTitle("目前沒有可同時繪製的前後測平均")
                self.group_chart_view.setChart(chart)
                return
            first_set = QBarSet("前測")
            second_set = QBarSet("後測")
            first_values = [float(item.first.mean) for item in plottable]
            second_values = [float(item.second.mean) for item in plottable]
            first_set.append(first_values)
            second_set.append(second_values)
            series = QBarSeries()
            series.append(first_set)
            series.append(second_set)
            chart.addSeries(series)
            axis_x = QBarCategoryAxis()
            axis_x.append([item.name for item in plottable])
            axis_y = QValueAxis()
            axis_y.setTitleText(comparison.metric.display_name)
            axis_y.setLabelFormat("%.3f")
            axis_y.setRange(0.0, max([*first_values, *second_values, 1e-6]) * 1.15)
            chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
            chart.legend().setVisible(True)
            self.group_chart_view.setChart(chart)

        def _render_subject_chart(self, rows, metric_name: str):
            paired = [
                row for row in rows if row.first_value is not None and row.second_value is not None
            ]
            chart = QChart()
            chart.setTitle(f"{metric_name}：受試者前後測")
            if not paired:
                chart.setTitle("目前範圍沒有有效的成對受試者資料")
                self.subject_chart_view.setChart(chart)
                return
            first_set = QBarSet("前測")
            second_set = QBarSet("後測")
            first_values = [float(row.first_value) for row in paired]
            second_values = [float(row.second_value) for row in paired]
            first_set.append(first_values)
            second_set.append(second_values)
            series = QBarSeries()
            series.append(first_set)
            series.append(second_set)
            chart.addSeries(series)
            axis_x = QBarCategoryAxis()
            axis_x.append([row.display_name for row in paired])
            axis_x.setLabelsAngle(-45)
            axis_y = QValueAxis()
            axis_y.setTitleText(metric_name)
            axis_y.setLabelFormat("%.3f")
            axis_y.setRange(0.0, max([*first_values, *second_values, 1e-6]) * 1.15)
            chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
            chart.legend().setVisible(True)
            self.subject_chart_view.setChart(chart)

        def render_segment_chart(self, *_args):
            comparison = self.current_comparison
            record_key = str(self.analysis_subject.currentData() or "")
            chart = QChart()
            if comparison is None or not record_key:
                chart.setTitle("尚未選擇受試者")
                self.segment_chart_view.setChart(chart)
                return
            row = next(
                (candidate for candidate in comparison.rows if candidate.pairing_key == record_key),
                None,
            )
            if row is None:
                chart.setTitle("找不到選取的受試者")
                self.segment_chart_view.setChart(chart)
                return

            all_points: list[tuple[int, float]] = []
            series_items = []
            for name, values in (
                ("前測", row.first_segment_values),
                ("後測", row.second_segment_values),
            ):
                points = [(index, float(value)) for index, value in values if value is not None]
                if not points:
                    continue
                series = QLineSeries()
                series.setName(name)
                for index, value in points:
                    series.append(float(index), value)
                chart.addSeries(series)
                series_items.append(series)
                all_points.extend(points)
            chart.setTitle(f"{row.display_name}：各段 {comparison.metric.display_name}")
            if not all_points:
                chart.setTitle(f"{row.display_name} 沒有可繪製的分段資料")
                self.segment_chart_view.setChart(chart)
                return
            x_values = [point[0] for point in all_points]
            y_values = [point[1] for point in all_points]
            axis_x = QValueAxis()
            axis_x.setTitleText("段落")
            axis_x.setLabelFormat("%d")
            axis_x.setRange(float(min(x_values)), float(max(x_values) if len(set(x_values)) > 1 else max(x_values) + 1))
            axis_x.setTickCount(max(2, min(11, max(x_values) - min(x_values) + 1)))
            axis_y = QValueAxis()
            axis_y.setTitleText(comparison.metric.display_name)
            axis_y.setLabelFormat("%.3f")
            minimum = min(0.0, min(y_values))
            maximum = max(y_values)
            axis_y.setRange(minimum, maximum * 1.15 if maximum > minimum else minimum + 1.0)
            chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            for series in series_items:
                series.attachAxis(axis_x)
                series.attachAxis(axis_y)
            chart.legend().setVisible(True)
            self.segment_chart_view.setChart(chart)

        def pick_reference(self):
            path, _ = QFileDialog.getOpenFileName(self, "選擇標準音檔", "", "Audio (*.wav *.mp3 *.m4a *.flac)")
            if path:
                self.reference.setText(path)

        def pick_transcript(self):
            path, _ = QFileDialog.getOpenFileName(self, "選擇分段文字檔", "", "Text (*.txt)")
            if path:
                self.transcript.setText(path)
                self.refresh_transcript_segment_count(show_error=True)

        def refresh_transcript_segment_count(self, show_error: bool = False) -> bool:
            path_text = self.transcript.text().strip()
            if not path_text:
                return False
            try:
                segments = read_transcript(Path(path_text))
            except Exception as exc:
                if show_error:
                    QMessageBox.warning(self, "分段文字格式錯誤", str(exc))
                return False
            self.segment_count.setValue(len(segments))
            self.status.setText(f"已讀取 {len(segments)} 個文字段落")
            return True

        def _source_paths(self) -> list[Path]:
            paths: list[Path] = []
            for index in range(self.source_directories.count()):
                item = self.source_directories.item(index)
                value = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
                if value:
                    paths.append(Path(value))
            return paths

        def _add_source_directories(self, raw_paths) -> None:
            existing = {str(path.resolve()).casefold() for path in self._source_paths()}
            for raw_path in raw_paths:
                path = Path(raw_path).resolve()
                key = str(path).casefold()
                if not path.is_dir() or key in existing:
                    continue
                item = QListWidgetItem(path.name or str(path))
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                item.setToolTip(str(path))
                self.source_directories.addItem(item)
                existing.add(key)
            self.status.setText(f"已選擇 {self.source_directories.count()} 個實驗資料夾")

        def pick_source(self):
            dialog = QFileDialog(self, "選擇同次實驗資料夾（Ctrl／Shift 可複選）")
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
            dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            for view_type in (QListView, QTreeView):
                for view in dialog.findChildren(view_type):
                    view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            if dialog.exec():
                self._add_source_directories(dialog.selectedFiles())

        def remove_selected_sources(self):
            for item in self.source_directories.selectedItems():
                self.source_directories.takeItem(self.source_directories.row(item))
            self.status.setText(f"已選擇 {self.source_directories.count()} 個實驗資料夾")

        def clear_selected_chopped(self):
            sources = self._source_paths()
            if not sources:
                QMessageBox.warning(self, "尚未選擇", "請先加入要清除 chopped 的資料夾。")
                return
            answer = QMessageBox.question(
                self,
                "確認清除切段快取",
                f"將刪除清單內 {len(sources)} 個資料夾中既有的 chopped 資料夾。\n"
                "原始音檔與 Excel 不會被刪除。是否繼續？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            try:
                removed = clear_chopped_directories(sources)
            except Exception as exc:
                QMessageBox.critical(self, "清除失敗", str(exc))
                return
            message = f"已清除 {len(removed)} 個 chopped 資料夾。"
            self.status.setText(message)
            self.log.appendPlainText(message)
            QMessageBox.information(self, "清除完成", message)

        def reset_interface(self):
            if self.process is not None:
                return

            self.reference.clear()
            self.transcript.clear()
            self.source_directories.clear()
            self.segment_count.setValue(1)
            self.threshold.setValue(0.30)
            self.segment_padding.setValue(1.0)
            self.emotion_model.setCurrentIndex(
                max(0, self.emotion_model.findData(DEFAULT_EMOTION_MODEL))
            )
            self.noise_reduction.setChecked(True)
            self.progress_bar.setValue(0)
            self.status.setText("就緒")
            self.log.clear()

            self.compare_pre_files.clear()
            self.compare_post_files.clear()
            self.compare_mode.setCurrentIndex(0)
            self.compare_metric.setCurrentIndex(0)
            self.compare_table.setRowCount(0)
            self.compare_status.setText("請分別加入一份以上的前測與後測報表。")
            self.compare_result.clear()
            self.compare_export_button.setEnabled(False)
            self.compare_refresh_button.setEnabled(False)
            self.comparison_first_run = None
            self.comparison_second_run = None
            self.current_comparison = None
            self.loaded_pairing_mode = None
            self.analysis_scope.setCurrentIndex(0)
            self._reset_interactive_analysis()

            self.process_stdout_buffer = ""
            self.process_error_tail = []
            self.process_failure_message = None
            self.analysis_outputs = []
            self.cancel_requested = False

        def start_analysis(self):
            if self.process is not None:
                return
            reference_text = self.reference.text().strip()
            transcript_text = self.transcript.text().strip()
            sources = self._source_paths()
            if not reference_text or not transcript_text or not sources:
                QMessageBox.warning(self, "資料不足", "請填寫標準音檔、分段文字檔並加入至少一個待分析資料夾。")
                return
            reference = Path(reference_text)
            transcript = Path(transcript_text)
            if not self.refresh_transcript_segment_count(show_error=True):
                return
            configs = [
                AnalysisConfig(
                    reference_audio=reference,
                    segment_count=self.segment_count.value(),
                    transcript_file=transcript,
                    source_directory=source,
                    match_threshold=self.threshold.value(),
                    output_excel=source / "emotion_analysis_result.xlsx",
                    emotion_model_name=str(self.emotion_model.currentData()),
                    segment_padding_seconds=self.segment_padding.value(),
                    noise_reduction_enabled=self.noise_reduction.isChecked(),
                )
                for source in sources
            ]
            try:
                self.job_file = create_job_file(configs)
            except Exception as exc:
                QMessageBox.critical(self, "無法啟動分析", str(exc))
                return
            self.log.clear()
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.reset_button.setEnabled(False)
            self.source_add_button.setEnabled(False)
            self.source_remove_button.setEnabled(False)
            self.source_clear_button.setEnabled(False)
            self.process_stdout_buffer = ""
            self.process_error_tail = []
            self.process_failure_message = None
            self.analysis_outputs = []
            self.cancel_requested = False
            self.process = QProcess(self)
            self.process.setProgram(sys.executable)
            self.process.setArguments(["-m", "emotion_analyzer.worker", str(self.job_file)])
            self.process.setWorkingDirectory(str(Path(__file__).resolve().parent.parent))
            process_environment = QProcessEnvironment.systemEnvironment()
            process_environment.insert("PYTHONUTF8", "1")
            self.process.setProcessEnvironment(process_environment)
            self.process.readyReadStandardOutput.connect(self.read_process_stdout)
            self.process.readyReadStandardError.connect(self.read_process_stderr)
            self.process.errorOccurred.connect(self.on_process_error)
            self.process.finished.connect(self.on_process_finished)
            self.status.setText(f"正在啟動批次分析，共 {len(configs)} 個資料夾…")
            self.process.start()

        def cancel_analysis(self):
            if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                self.cancel_requested = True
                self.status.setText("正在取消，請稍候…")
                self.process.terminate()
                QTimer.singleShot(3000, self.kill_process_if_running)

        def kill_process_if_running(self):
            if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()

        def read_process_stdout(self):
            if not self.process:
                return
            chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self.process_stdout_buffer += chunk
            while "\n" in self.process_stdout_buffer:
                line, self.process_stdout_buffer = self.process_stdout_buffer.split("\n", 1)
                self.handle_process_line(line.rstrip("\r"))

        def read_process_stderr(self):
            if not self.process:
                return
            chunk = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
            lines = [line for line in chunk.splitlines() if line.strip()]
            self.process_error_tail = (self.process_error_tail + lines)[-12:]

        def handle_process_line(self, line: str):
            event = decode_event(line)
            if event is None:
                return
            event_name = event["event"]
            if event_name == "progress":
                value = int(event.get("value", 0))
                message = str(event.get("message", ""))
                self.progress_bar.setValue(value)
                self.status.setText(message)
                self.log.appendPlainText(message)
            elif event_name == "failed":
                self.process_failure_message = str(event.get("message", "分析失敗"))
            elif event_name == "finished":
                raw_outputs = event.get("outputs")
                if isinstance(raw_outputs, list):
                    self.analysis_outputs = [str(output) for output in raw_outputs if str(output)]
                else:
                    output = str(event.get("output", ""))
                    self.analysis_outputs = [output] if output else []

        def on_process_error(self, _error):
            if self.process:
                self.process_failure_message = self.process.errorString()

        def on_process_finished(self, exit_code: int, _exit_status):
            if self.process:
                self.read_process_stdout()
                if self.process_stdout_buffer:
                    self.handle_process_line(self.process_stdout_buffer.rstrip("\r\n"))
                self.read_process_stderr()
                self.process.deleteLater()
            self.process = None
            if self.job_file:
                try:
                    self.job_file.unlink(missing_ok=True)
                except OSError:
                    pass
            self.job_file = None
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.reset_button.setEnabled(True)
            self.source_add_button.setEnabled(True)
            self.source_remove_button.setEnabled(True)
            self.source_clear_button.setEnabled(True)

            if self.cancel_requested:
                self.status.setText("分析已取消")
                self.log.appendPlainText("分析已取消")
                return
            if exit_code == 0 and self.analysis_outputs:
                self.progress_bar.setValue(100)
                self.status.setText("分析完成")
                output_text = "\n".join(self.analysis_outputs)
                self.log.appendPlainText(f"已輸出 {len(self.analysis_outputs)} 份 Excel：\n{output_text}")
                QMessageBox.information(
                    self,
                    "完成",
                    f"分析完成，共輸出 {len(self.analysis_outputs)} 份 Excel：\n{output_text}",
                )
                return

            details = self.process_failure_message or "分析行程意外關閉"
            if self.process_error_tail:
                details += "\n\n最後的執行訊息：\n" + "\n".join(self.process_error_tail)
            self.status.setText("分析失敗")
            self.log.appendPlainText(details)
            QMessageBox.critical(self, "分析失敗", details)

        def closeEvent(self, event):
            if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()
                self.process.waitForFinished(2000)
            if self.job_file:
                try:
                    self.job_file.unlink(missing_ok=True)
                except OSError:
                    pass
            super().closeEvent(event)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
