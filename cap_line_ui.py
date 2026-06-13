#!/usr/bin/env python3
"""
PyQt operator UI for the two-class cap inspection runtime.
"""

from __future__ import annotations

from contextlib import closing
import csv
import json
import os
import queue
import sqlite3
import threading
import traceback
from datetime import datetime
from typing import Callable

import cap_line_runtime
from cap_line_runtime import DetectionHistoryRecord, TimingLogRecord


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "data", "cap_line_history.sqlite3")
DEFAULT_MODEL = cap_line_runtime.DEFAULT_MODEL
DEFAULT_TIMING_LOG_DIR = cap_line_runtime.DEFAULT_TIMING_LOG_DIR
EVENT_LIMIT = 100
TIMING_LOG_LIMIT = 100
TRIGGER_PIN_LABEL = "Trigger GPIO (CVM)"


class HistoryRepository:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cap_line_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        recorded_at TEXT NOT NULL,
                        recorded_date TEXT NOT NULL,
                        recorded_year INTEGER NOT NULL,
                        recorded_month TEXT NOT NULL,
                        recorded_iso_year INTEGER NOT NULL,
                        recorded_iso_week INTEGER NOT NULL,
                        runtime_event_id INTEGER NOT NULL,
                        result TEXT NOT NULL,
                        final_class_name TEXT,
                        final_score REAL,
                        decision_source TEXT NOT NULL,
                        camera_labels_json TEXT NOT NULL,
                        camera_votes_json TEXT NOT NULL,
                        anchor_time TEXT,
                        trigger_delay_s REAL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_cap_line_history_recorded_at
                    ON cap_line_history (recorded_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_cap_line_history_recorded_date
                    ON cap_line_history (recorded_date DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_cap_line_history_recorded_month
                    ON cap_line_history (recorded_month DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_cap_line_history_recorded_year
                    ON cap_line_history (recorded_year DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_cap_line_history_recorded_iso_week
                    ON cap_line_history (recorded_iso_year DESC, recorded_iso_week DESC)
                    """
                )

    def insert_record(self, record: DetectionHistoryRecord) -> None:
        recorded_time = datetime.fromisoformat(record.recorded_at)
        iso_year, iso_week, _ = recorded_time.isocalendar()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO cap_line_history (
                        recorded_at,
                        recorded_date,
                        recorded_year,
                        recorded_month,
                        recorded_iso_year,
                        recorded_iso_week,
                        runtime_event_id,
                        result,
                        final_class_name,
                        final_score,
                        decision_source,
                        camera_labels_json,
                        camera_votes_json,
                        anchor_time,
                        trigger_delay_s
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.recorded_at,
                        recorded_time.strftime("%Y-%m-%d"),
                        recorded_time.year,
                        recorded_time.strftime("%Y-%m"),
                        iso_year,
                        iso_week,
                        record.runtime_event_id,
                        record.result,
                        record.final_class_name,
                        record.final_score,
                        record.decision_source,
                        json.dumps(record.camera_labels),
                        json.dumps(record.camera_votes),
                        record.anchor_time,
                        record.trigger_delay_s,
                    ),
                )

    def fetch_events(self, limit: int = EVENT_LIMIT) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    recorded_at,
                    runtime_event_id,
                    result,
                    final_class_name,
                    final_score,
                    decision_source,
                    camera_labels_json,
                    camera_votes_json,
                    anchor_time,
                    trigger_delay_s
                FROM cap_line_history
                ORDER BY recorded_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        return [
            {
                "recorded_at": row["recorded_at"],
                "runtime_event_id": row["runtime_event_id"],
                "result": row["result"],
                "final_class_name": row["final_class_name"],
                "final_score": row["final_score"],
                "decision_source": row["decision_source"],
                "camera_labels": json.loads(row["camera_labels_json"]),
                "camera_votes": json.loads(row["camera_votes_json"]),
                "anchor_time": row["anchor_time"],
                "trigger_delay_s": row["trigger_delay_s"],
            }
            for row in rows
        ]

    def _fetch_grouped_stats(
        self,
        *,
        period_select: str,
        group_by: str,
        order_by: str,
    ) -> list[dict[str, object]]:
        query = f"""
            SELECT
                {period_select} AS period,
                COUNT(*) AS total_items,
                SUM(CASE WHEN result = 'trigger' THEN 1 ELSE 0 END) AS reject_triggers,
                SUM(CASE WHEN result = 'skip' AND final_class_name = 'undefected' THEN 1 ELSE 0 END) AS undefected_wins,
                SUM(CASE WHEN result = 'trigger' AND final_class_name = 'dirt_defect' THEN 1 ELSE 0 END) AS dirt_defects
            FROM cap_line_history
            GROUP BY {group_by}
            ORDER BY {order_by}
        """

        with closing(self._connect()) as connection:
            rows = connection.execute(query).fetchall()

        stats: list[dict[str, object]] = []
        for row in rows:
            total_items = int(row["total_items"])
            reject_triggers = int(row["reject_triggers"])
            stats.append(
                {
                    "period": row["period"],
                    "total_items": total_items,
                    "reject_triggers": reject_triggers,
                    "undefected_wins": int(row["undefected_wins"]),
                    "dirt_defects": int(row["dirt_defects"]),
                    "reject_rate": round(
                        (100.0 * reject_triggers / total_items) if total_items else 0.0,
                        2,
                    ),
                }
            )
        return stats

    def fetch_daily_stats(self) -> list[dict[str, object]]:
        return self._fetch_grouped_stats(
            period_select="recorded_date",
            group_by="recorded_date",
            order_by="recorded_date DESC",
        )

    def fetch_weekly_stats(self) -> list[dict[str, object]]:
        return self._fetch_grouped_stats(
            period_select="printf('%04d-W%02d', recorded_iso_year, recorded_iso_week)",
            group_by="recorded_iso_year, recorded_iso_week",
            order_by="recorded_iso_year DESC, recorded_iso_week DESC",
        )

    def fetch_monthly_stats(self) -> list[dict[str, object]]:
        return self._fetch_grouped_stats(
            period_select="recorded_month",
            group_by="recorded_month",
            order_by="recorded_month DESC",
        )

    def fetch_yearly_stats(self) -> list[dict[str, object]]:
        return self._fetch_grouped_stats(
            period_select="CAST(recorded_year AS TEXT)",
            group_by="recorded_year",
            order_by="recorded_year DESC",
        )


def read_recent_timing_logs(directory: str, limit: int = TIMING_LOG_LIMIT) -> list[dict[str, str]]:
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return []

    rows: list[dict[str, str]] = []
    filenames = sorted(
        (
            entry_name
            for entry_name in os.listdir(directory)
            if entry_name.lower().endswith(".csv")
        ),
        reverse=True,
    )
    for entry_name in filenames:
        file_path = os.path.join(directory, entry_name)
        try:
            with open(file_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows.extend(reader)
        except OSError:
            continue
        if len(rows) >= limit * 2:
            break

    rows.sort(key=lambda row: row.get("recorded_at", ""), reverse=True)
    return rows[:limit]


def create_gui_args():
    args = cap_line_runtime.parse_args([])
    args.model = DEFAULT_MODEL
    args.no_display = True
    if os.name == "nt":
        args.simulate_gpio = True
    return args


class DetectionAppController:
    def __init__(
        self,
        repository: HistoryRepository,
        *,
        detector_runner: Callable[..., None] = cap_line_runtime.run_detection,
        args_factory: Callable[[], object] = create_gui_args,
    ):
        self.repository = repository
        self.detector_runner = detector_runner
        self.args_factory = args_factory
        self.status_text = "Stopped"
        self.is_running = False
        self.worker_thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self._message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._preview_lock = threading.Lock()
        self._latest_preview = None

    def start(self) -> bool:
        if self.is_running:
            return False

        args = self.args_factory()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._worker_main,
            args=(args,),
            name="cap-line-ui-worker",
            daemon=True,
        )
        self._message_queue = queue.Queue()
        with self._preview_lock:
            self._latest_preview = None
        self.is_running = True
        self.status_text = "Running"
        self.worker_thread.start()
        return True

    def stop(self) -> bool:
        if not self.is_running or self.stop_event is None:
            return False
        self.stop_event.set()
        self.status_text = "Stopping..."
        return True

    def _worker_main(self, args) -> None:
        try:
            self.detector_runner(
                args,
                stop_event=self.stop_event,
                preview_callback=self._store_preview,
                history_callback=self._queue_history_record,
                timing_log_callback=self._queue_timing_record,
                log_fn=print,
            )
        except Exception as exc:
            traceback.print_exc()
            self._message_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self._message_queue.put(("stopped", None))

    def _store_preview(self, preview_frame) -> None:
        with self._preview_lock:
            self._latest_preview = preview_frame.copy()

    def _queue_history_record(self, record: DetectionHistoryRecord) -> None:
        self._message_queue.put(("history", record))

    def _queue_timing_record(self, record: TimingLogRecord) -> None:
        self._message_queue.put(("timing", record))

    def drain_messages(self) -> dict[str, object]:
        history_records: list[DetectionHistoryRecord] = []
        timing_records: list[TimingLogRecord] = []
        latest_error = None
        stopped = False

        while True:
            try:
                kind, payload = self._message_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "history":
                self.repository.insert_record(payload)
                history_records.append(payload)
            elif kind == "timing":
                timing_records.append(payload)
            elif kind == "error":
                latest_error = str(payload)
                self.status_text = f"Error: {latest_error}"
            elif kind == "stopped":
                stopped = True
                self.is_running = False
                self.worker_thread = None
                self.stop_event = None
                if latest_error is None and not self.status_text.startswith("Error:"):
                    self.status_text = "Stopped"

        with self._preview_lock:
            latest_preview = self._latest_preview
            self._latest_preview = None

        return {
            "history_records": history_records,
            "timing_records": timing_records,
            "latest_preview": latest_preview,
            "error": latest_error,
            "stopped": stopped,
        }


try:
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    PYQT_AVAILABLE = True
except ModuleNotFoundError:
    PYQT_AVAILABLE = False


if PYQT_AVAILABLE:
    def _format_timestamp(value: str | None) -> str:
        if not value:
            return "-"
        try:
            return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return value


    def _format_float(value: object, digits: int = 3) -> str:
        if value in (None, ""):
            return "-"
        return f"{float(value):.{digits}f}"


    def _safe_item(value: object) -> QTableWidgetItem:
        return QTableWidgetItem("" if value is None else str(value))


    class DetectionApp(QWidget):
        def __init__(
            self,
            *,
            repository: HistoryRepository | None = None,
            controller: DetectionAppController | None = None,
        ):
            super().__init__()
            self.repository = repository or HistoryRepository()
            self.controller = controller or DetectionAppController(self.repository)
            self.controller.args_factory = self._build_runtime_args
            self._latest_preview_image = None
            self._closing_after_stop = False
            self.current_stats_period = "daily"
            self.session_counts = {
                "total_items": 0,
                "reject_triggers": 0,
                "undefected_wins": 0,
                "dirt_defects": 0,
            }

            self.setWindowTitle("Cap Line Inspector")
            self.resize(1440, 920)
            self._build_ui()
            self._load_history_tables()
            self._load_timing_table()
            self._sync_controls()

            self.poll_timer = QTimer(self)
            self.poll_timer.setInterval(150)
            self.poll_timer.timeout.connect(self._poll_controller)
            self.poll_timer.start()

        def _build_ui(self) -> None:
            self.setStyleSheet(
                """
                QWidget {
                    background: #0e1820;
                    color: #f4f7fb;
                    font-size: 10pt;
                }
                QGroupBox {
                    border: 1px solid #264150;
                    border-radius: 10px;
                    margin-top: 10px;
                    padding-top: 12px;
                    background: #14232e;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 4px;
                }
                QPushButton {
                    min-height: 36px;
                    padding: 0 14px;
                    border-radius: 10px;
                    border: 1px solid #2f5368;
                    background: #1c3240;
                }
                QPushButton:disabled {
                    color: #7f94a3;
                    background: #19242b;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTableWidget {
                    background: #101a21;
                    border: 1px solid #2b4454;
                    border-radius: 8px;
                    padding: 4px;
                }
                QLabel#StatusValue {
                    font-size: 16pt;
                    font-weight: 700;
                    color: #6ef0c2;
                }
                QLabel#MetricValue {
                    font-size: 14pt;
                    font-weight: 700;
                    color: #ffd16b;
                }
                QLabel#MetricTitle {
                    color: #8ca4b6;
                }
                QLabel#PreviewSurface {
                    background: #081118;
                    border: 1px solid #294758;
                    border-radius: 12px;
                    padding: 8px;
                }
                QHeaderView::section {
                    background: #1a2c38;
                    color: #f4f7fb;
                    padding: 6px;
                    border: none;
                    border-right: 1px solid #264150;
                }
                """
            )

            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(16, 16, 16, 16)
            root_layout.setSpacing(12)

            header_layout = QHBoxLayout()
            title = QLabel("Cap Line Inspector")
            title.setStyleSheet("font-size: 22pt; font-weight: 700; color: #f7fbff;")
            header_layout.addWidget(title)
            header_layout.addStretch(1)
            self.status_value = QLabel("Stopped")
            self.status_value.setObjectName("StatusValue")
            header_layout.addWidget(self.status_value)
            root_layout.addLayout(header_layout)

            self.tabs = QTabWidget()
            self.live_tab = QWidget()
            self.history_tab = QWidget()
            self.tabs.addTab(self.live_tab, "Live")
            self.tabs.addTab(self.history_tab, "History")
            root_layout.addWidget(self.tabs)

            self._build_live_tab()
            self._build_history_tab()

        def _build_live_tab(self) -> None:
            layout = QVBoxLayout(self.live_tab)
            layout.setSpacing(12)

            top_splitter = QSplitter(Qt.Orientation.Horizontal)
            layout.addWidget(top_splitter, 1)

            preview_group = QGroupBox("Preview")
            preview_layout = QVBoxLayout(preview_group)
            self.preview_label = QLabel("Waiting for preview frames")
            self.preview_label.setObjectName("PreviewSurface")
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setMinimumSize(760, 420)
            preview_layout.addWidget(self.preview_label)
            top_splitter.addWidget(preview_group)

            side_panel = QWidget()
            side_layout = QVBoxLayout(side_panel)
            side_layout.setSpacing(12)
            top_splitter.addWidget(side_panel)
            top_splitter.setStretchFactor(0, 3)
            top_splitter.setStretchFactor(1, 2)

            control_group = QGroupBox("Controls")
            control_layout = QVBoxLayout(control_group)
            button_row = QHBoxLayout()
            self.start_button = QPushButton("Start")
            self.stop_button = QPushButton("Stop")
            self.start_button.clicked.connect(self._start_detection)
            self.stop_button.clicked.connect(self._stop_detection)
            button_row.addWidget(self.start_button)
            button_row.addWidget(self.stop_button)
            control_layout.addLayout(button_row)
            side_layout.addWidget(control_group)

            metrics_group = QGroupBox("Session")
            metrics_layout = QGridLayout(metrics_group)
            self.metric_labels: dict[str, QLabel] = {}
            metric_defs = [
                ("Total Items", "total_items"),
                ("Reject Triggers", "reject_triggers"),
                ("Undefected Wins", "undefected_wins"),
                ("Dirt Defects", "dirt_defects"),
                ("Reject Rate", "reject_rate"),
            ]
            for index, (label_text, key) in enumerate(metric_defs):
                title = QLabel(label_text)
                title.setObjectName("MetricTitle")
                value = QLabel("0")
                value.setObjectName("MetricValue")
                self.metric_labels[key] = value
                metrics_layout.addWidget(title, index, 0)
                metrics_layout.addWidget(value, index, 1)
            side_layout.addWidget(metrics_group)

            config_group = QGroupBox("Calibration")
            config_form = QFormLayout(config_group)
            self.model_input = QLineEdit(DEFAULT_MODEL)
            self.cam0_input = QLineEdit("0")
            self.cam1_input = QLineEdit("2")
            self.width_spin = QSpinBox()
            self.width_spin.setRange(160, 4096)
            self.width_spin.setValue(640)
            self.height_spin = QSpinBox()
            self.height_spin.setRange(120, 4096)
            self.height_spin.setValue(480)
            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(1, 240)
            self.fps_spin.setValue(15)
            self.exposure_spin = QSpinBox()
            self.exposure_spin.setRange(1, 10000)
            self.exposure_spin.setValue(8)
            self.conf_spin = QDoubleSpinBox()
            self.conf_spin.setRange(0.0, 1.0)
            self.conf_spin.setSingleStep(0.01)
            self.conf_spin.setDecimals(3)
            self.conf_spin.setValue(cap_line_runtime.DEFAULT_CONFIDENCE)
            self.trigger_pin_input = QLineEdit(cap_line_runtime.DEFAULT_TRIGGER_PIN)
            self.trigger_duration_spin = QDoubleSpinBox()
            self.trigger_duration_spin.setRange(0.01, 10.0)
            self.trigger_duration_spin.setDecimals(3)
            self.trigger_duration_spin.setValue(0.3)
            self.trigger_gap_spin = QDoubleSpinBox()
            self.trigger_gap_spin.setRange(0.0, 10.0)
            self.trigger_gap_spin.setDecimals(3)
            self.trigger_gap_spin.setValue(0.0)
            self.timing_camera_combo = QComboBox()
            self.timing_camera_combo.addItems(["0", "1"])
            self.anchor_axis_combo = QComboBox()
            self.anchor_axis_combo.addItems(["x", "y"])
            self.anchor_line_spin = QDoubleSpinBox()
            self.anchor_line_spin.setRange(0.0, 1.0)
            self.anchor_line_spin.setDecimals(3)
            self.anchor_line_spin.setSingleStep(0.05)
            self.anchor_line_spin.setValue(0.5)
            self.defect_min_score_spin = QDoubleSpinBox()
            self.defect_min_score_spin.setRange(0.0, 1.0)
            self.defect_min_score_spin.setDecimals(3)
            self.defect_min_score_spin.setSingleStep(0.01)
            self.defect_min_score_spin.setValue(cap_line_runtime.DEFAULT_DEFECT_MIN_SCORE)
            self.defect_margin_spin = QDoubleSpinBox()
            self.defect_margin_spin.setRange(0.0, 1.0)
            self.defect_margin_spin.setDecimals(3)
            self.defect_margin_spin.setSingleStep(0.01)
            self.defect_margin_spin.setValue(cap_line_runtime.DEFAULT_DEFECT_MARGIN)
            self.single_camera_defect_score_spin = QDoubleSpinBox()
            self.single_camera_defect_score_spin.setRange(0.0, 1.0)
            self.single_camera_defect_score_spin.setDecimals(3)
            self.single_camera_defect_score_spin.setSingleStep(0.01)
            self.single_camera_defect_score_spin.setValue(
                cap_line_runtime.DEFAULT_SINGLE_CAMERA_DEFECT_SCORE
            )
            self.finalize_quiet_spin = QDoubleSpinBox()
            self.finalize_quiet_spin.setRange(0.0, 5000.0)
            self.finalize_quiet_spin.setDecimals(1)
            self.finalize_quiet_spin.setSingleStep(5.0)
            self.finalize_quiet_spin.setValue(cap_line_runtime.DEFAULT_FINALIZE_QUIET_MS)
            self.nozzle_distance_spin = QDoubleSpinBox()
            self.nozzle_distance_spin.setRange(0.0, 5000.0)
            self.nozzle_distance_spin.setDecimals(3)
            self.nozzle_distance_spin.setValue(cap_line_runtime.DEFAULT_NOZZLE_DISTANCE_MM)
            self.belt_speed_spin = QDoubleSpinBox()
            self.belt_speed_spin.setRange(0.001, 5000.0)
            self.belt_speed_spin.setDecimals(3)
            self.belt_speed_spin.setValue(cap_line_runtime.DEFAULT_BELT_SPEED_MM_PER_S)
            self.trigger_offset_spin = QDoubleSpinBox()
            self.trigger_offset_spin.setRange(-5.0, 5.0)
            self.trigger_offset_spin.setDecimals(3)
            self.trigger_offset_spin.setValue(cap_line_runtime.DEFAULT_TRIGGER_OFFSET_S)
            self.latency_compensation_spin = QDoubleSpinBox()
            self.latency_compensation_spin.setRange(0.0, 5000.0)
            self.latency_compensation_spin.setDecimals(1)
            self.latency_compensation_spin.setSingleStep(5.0)
            self.latency_compensation_spin.setValue(
                cap_line_runtime.DEFAULT_LATENCY_COMPENSATION_MS
            )
            self.timing_log_dir_input = QLineEdit(DEFAULT_TIMING_LOG_DIR)
            self.review_dir_input = QLineEdit(cap_line_runtime.DEFAULT_REVIEW_DIR)
            self.simulate_gpio_checkbox = QCheckBox("Simulate GPIO")
            self.simulate_gpio_checkbox.setChecked(os.name == "nt")

            config_form.addRow("Model", self.model_input)
            config_form.addRow("Camera 0", self.cam0_input)
            config_form.addRow("Camera 1", self.cam1_input)
            config_form.addRow("Width", self.width_spin)
            config_form.addRow("Height", self.height_spin)
            config_form.addRow("FPS", self.fps_spin)
            config_form.addRow("Exposure", self.exposure_spin)
            config_form.addRow("Confidence", self.conf_spin)
            config_form.addRow(TRIGGER_PIN_LABEL, self.trigger_pin_input)
            config_form.addRow("Trigger Duration", self.trigger_duration_spin)
            config_form.addRow("Trigger Min Gap", self.trigger_gap_spin)
            config_form.addRow("Timing Camera", self.timing_camera_combo)
            config_form.addRow("Anchor Axis", self.anchor_axis_combo)
            config_form.addRow("Anchor Line Ratio", self.anchor_line_spin)
            config_form.addRow("Defect Min Score", self.defect_min_score_spin)
            config_form.addRow("Defect Margin", self.defect_margin_spin)
            config_form.addRow("Single-Camera Defect Score", self.single_camera_defect_score_spin)
            config_form.addRow("Finalize Quiet ms", self.finalize_quiet_spin)
            config_form.addRow("Nozzle Distance mm", self.nozzle_distance_spin)
            config_form.addRow("Belt Speed mm/s", self.belt_speed_spin)
            config_form.addRow("Trigger Offset s", self.trigger_offset_spin)
            config_form.addRow("Latency Compensation ms", self.latency_compensation_spin)
            config_form.addRow("Timing Log Dir", self.timing_log_dir_input)
            config_form.addRow("Review Capture Dir", self.review_dir_input)
            config_form.addRow("", self.simulate_gpio_checkbox)
            side_layout.addWidget(config_group, 1)

            recent_group = QGroupBox("Recent Events")
            recent_layout = QVBoxLayout(recent_group)
            self.recent_events_table = QTableWidget(0, 6)
            self.recent_events_table.setHorizontalHeaderLabels(
                ["Recorded", "Event", "Result", "Class", "Score", "Source"]
            )
            self.recent_events_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.recent_events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.recent_events_table.verticalHeader().setVisible(False)
            recent_layout.addWidget(self.recent_events_table)
            layout.addWidget(recent_group, 1)

            timing_group = QGroupBox("Timing Review")
            timing_layout = QVBoxLayout(timing_group)
            self.timing_log_path_label = QLabel(DEFAULT_TIMING_LOG_DIR)
            timing_layout.addWidget(self.timing_log_path_label)
            self.timing_table = QTableWidget(0, 8)
            self.timing_table.setHorizontalHeaderLabels(
                [
                    "Recorded",
                    "Event",
                    "Result",
                    "Class",
                    "Anchor",
                    "Requested",
                    "Trigger On",
                    "Late ms",
                ]
            )
            self.timing_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.timing_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.timing_table.verticalHeader().setVisible(False)
            timing_layout.addWidget(self.timing_table)
            layout.addWidget(timing_group, 1)

        def _build_history_tab(self) -> None:
            layout = QVBoxLayout(self.history_tab)
            layout.setSpacing(12)

            stats_group = QGroupBox("Aggregates")
            stats_layout = QVBoxLayout(stats_group)
            period_row = QHBoxLayout()
            self.stats_period_combo = QComboBox()
            self.stats_period_combo.addItems(["daily", "weekly", "monthly", "yearly"])
            self.stats_period_combo.currentTextChanged.connect(self._change_stats_period)
            period_row.addWidget(QLabel("Period"))
            period_row.addWidget(self.stats_period_combo)
            period_row.addStretch(1)
            stats_layout.addLayout(period_row)

            self.stats_table = QTableWidget(0, 5)
            self.stats_table.setHorizontalHeaderLabels(
                ["Period", "Total", "Rejects", "Undefected", "Reject Rate"]
            )
            self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.stats_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.stats_table.verticalHeader().setVisible(False)
            stats_layout.addWidget(self.stats_table)
            layout.addWidget(stats_group, 1)

            history_group = QGroupBox("Stored History")
            history_layout = QVBoxLayout(history_group)
            self.history_events_table = QTableWidget(0, 7)
            self.history_events_table.setHorizontalHeaderLabels(
                ["Recorded", "Event", "Result", "Class", "Score", "Source", "Anchor"]
            )
            self.history_events_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.history_events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.history_events_table.verticalHeader().setVisible(False)
            history_layout.addWidget(self.history_events_table)
            layout.addWidget(history_group, 1)

        def _build_runtime_args(self):
            args = cap_line_runtime.parse_args([])
            args.no_display = True
            args.model = self.model_input.text().strip() or DEFAULT_MODEL
            args.cams = [self.cam0_input.text().strip() or "0", self.cam1_input.text().strip() or "2"]
            args.res = [self.width_spin.value(), self.height_spin.value()]
            args.fps = self.fps_spin.value()
            args.exposure = self.exposure_spin.value()
            args.conf = self.conf_spin.value()
            args.trigger_pin = (
                self.trigger_pin_input.text().strip()
                or cap_line_runtime.DEFAULT_TRIGGER_PIN
            )
            args.trigger_duration = self.trigger_duration_spin.value()
            args.trigger_min_gap = self.trigger_gap_spin.value()
            args.timing_camera = int(self.timing_camera_combo.currentText())
            args.anchor_axis = self.anchor_axis_combo.currentText()
            args.anchor_line_ratio = self.anchor_line_spin.value()
            args.defect_min_score = self.defect_min_score_spin.value()
            args.defect_margin = self.defect_margin_spin.value()
            args.single_camera_defect_score = self.single_camera_defect_score_spin.value()
            args.finalize_quiet_ms = self.finalize_quiet_spin.value()
            args.nozzle_distance_mm = self.nozzle_distance_spin.value()
            args.belt_speed_mm_per_s = self.belt_speed_spin.value()
            args.trigger_offset_s = self.trigger_offset_spin.value()
            args.latency_compensation_ms = self.latency_compensation_spin.value()
            args.timing_log_dir = self.timing_log_dir_input.text().strip() or DEFAULT_TIMING_LOG_DIR
            args.review_dir = self.review_dir_input.text().strip() or cap_line_runtime.DEFAULT_REVIEW_DIR
            args.simulate_gpio = self.simulate_gpio_checkbox.isChecked()
            return args

        def _start_detection(self) -> None:
            if self.controller.start():
                self._reset_session_counts()
                self._sync_controls()

        def _stop_detection(self) -> None:
            if self.controller.stop():
                self._sync_controls()

        def _reset_session_counts(self) -> None:
            self.session_counts = {
                "total_items": 0,
                "reject_triggers": 0,
                "undefected_wins": 0,
                "dirt_defects": 0,
            }
            self._update_session_metrics()

        def _update_session_metrics(self) -> None:
            total_items = self.session_counts["total_items"]
            reject_triggers = self.session_counts["reject_triggers"]
            reject_rate = (100.0 * reject_triggers / total_items) if total_items else 0.0
            self.metric_labels["total_items"].setText(str(total_items))
            self.metric_labels["reject_triggers"].setText(str(reject_triggers))
            self.metric_labels["undefected_wins"].setText(str(self.session_counts["undefected_wins"]))
            self.metric_labels["dirt_defects"].setText(str(self.session_counts["dirt_defects"]))
            self.metric_labels["reject_rate"].setText(f"{reject_rate:.2f}%")

        def _sync_controls(self) -> None:
            self.status_value.setText(self.controller.status_text)
            self.start_button.setEnabled(not self.controller.is_running)
            self.stop_button.setEnabled(self.controller.is_running)

        def _poll_controller(self) -> None:
            changes = self.controller.drain_messages()
            if changes["latest_preview"] is not None:
                self._update_preview(changes["latest_preview"])

            history_records = changes["history_records"]
            if history_records:
                for record in history_records:
                    self._apply_history_record(record)
                self._load_history_tables()

            if changes["timing_records"]:
                self._load_timing_table()

            if changes["error"] is not None or changes["stopped"]:
                self._sync_controls()

            if self._closing_after_stop and not self.controller.is_running:
                self._closing_after_stop = False
                self.close()

        def _apply_history_record(self, record: DetectionHistoryRecord) -> None:
            self.session_counts["total_items"] += 1
            if record.result == "trigger":
                self.session_counts["reject_triggers"] += 1
            if record.final_class_name == "undefected" and record.result == "skip":
                self.session_counts["undefected_wins"] += 1
            if record.final_class_name == "dirt_defect" and record.result == "trigger":
                self.session_counts["dirt_defects"] += 1
            self._update_session_metrics()

        def _update_preview(self, preview_frame) -> None:
            import cv2

            rgb_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_frame.shape
            bytes_per_line = channels * width
            image = QImage(
                rgb_frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()
            self._latest_preview_image = image
            pixmap = QPixmap.fromImage(image)
            self.preview_label.setText("")
            self.preview_label.setPixmap(
                pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            if self._latest_preview_image is not None:
                pixmap = QPixmap.fromImage(self._latest_preview_image)
                self.preview_label.setPixmap(
                    pixmap.scaled(
                        self.preview_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

        def _change_stats_period(self, period: str) -> None:
            self.current_stats_period = period
            self._load_stats_table()

        def _load_history_tables(self) -> None:
            events = self.repository.fetch_events(limit=EVENT_LIMIT)
            self._populate_event_table(self.recent_events_table, events, include_anchor=False)
            self._populate_event_table(self.history_events_table, events, include_anchor=True)
            self._load_stats_table()

        def _load_stats_table(self) -> None:
            fetchers = {
                "daily": self.repository.fetch_daily_stats,
                "weekly": self.repository.fetch_weekly_stats,
                "monthly": self.repository.fetch_monthly_stats,
                "yearly": self.repository.fetch_yearly_stats,
            }
            rows = fetchers[self.current_stats_period]()
            self.stats_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                values = [
                    row["period"],
                    row["total_items"],
                    row["reject_triggers"],
                    row["undefected_wins"],
                    f"{float(row['reject_rate']):.2f}%",
                ]
                for column_index, value in enumerate(values):
                    self.stats_table.setItem(row_index, column_index, _safe_item(value))
            self.stats_table.resizeColumnsToContents()

        def _populate_event_table(
            self,
            table: QTableWidget,
            events: list[dict[str, object]],
            *,
            include_anchor: bool,
        ) -> None:
            table.setRowCount(len(events))
            for row_index, event in enumerate(events):
                values = [
                    _format_timestamp(str(event["recorded_at"])),
                    event["runtime_event_id"],
                    str(event["result"]).title(),
                    event["final_class_name"] or "-",
                    _format_float(event["final_score"]),
                    event["decision_source"],
                ]
                if include_anchor:
                    values.append(_format_timestamp(event["anchor_time"]))
                for column_index, value in enumerate(values):
                    table.setItem(row_index, column_index, _safe_item(value))
            table.resizeColumnsToContents()

        def _load_timing_table(self) -> None:
            directory = self.timing_log_dir_input.text().strip() or DEFAULT_TIMING_LOG_DIR
            self.timing_log_path_label.setText(directory)
            rows = read_recent_timing_logs(directory, limit=TIMING_LOG_LIMIT)
            self.timing_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                values = [
                    _format_timestamp(row.get("recorded_at")),
                    row.get("event_id", ""),
                    row.get("final_result", "").title(),
                    row.get("final_class", "") or "-",
                    _format_timestamp(row.get("anchor_time")),
                    _format_timestamp(row.get("requested_fire_time")),
                    _format_timestamp(row.get("trigger_on_time")),
                    _format_float(row.get("scheduler_late_ms")),
                ]
                for column_index, value in enumerate(values):
                    self.timing_table.setItem(row_index, column_index, _safe_item(value))
            self.timing_table.resizeColumnsToContents()

        def closeEvent(self, event: QCloseEvent) -> None:
            if self.controller.is_running:
                self.controller.stop()
                self._closing_after_stop = True
                self._sync_controls()
                event.ignore()
                return
            self.poll_timer.stop()
            super().closeEvent(event)


    def main() -> None:
        app = QApplication([])
        window = DetectionApp()
        window.show()
        app.exec()


else:
    class DetectionApp:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyQt6 is required to use cap_line_ui.py")


    def main() -> None:
        raise RuntimeError("PyQt6 is required to run cap_line_ui.py")


if __name__ == "__main__":
    main()
