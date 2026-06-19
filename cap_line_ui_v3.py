#!/usr/bin/env python3
"""Standalone PyQt operator UI for the V3 cap-line runtime."""

from __future__ import annotations

from contextlib import closing
import json
import os
import queue
import sqlite3
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from cap_line_runtime_v3 import RuntimeCallbacks, RuntimeConfig, run_detection


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(SCRIPT_DIR / "data" / "cap_line_history_v3.sqlite3")
DEFAULT_SETTINGS_PATH = str(SCRIPT_DIR / "data" / "cap_line_ui_v3_settings.json")
EVENT_LIMIT = 100
TIMING_LOG_LIMIT = 100
LIVE_POLL_INTERVAL_MS = 16
TRIGGER_PIN_LABEL = "Trigger GPIO09 (BOARD pin 7)"
CONFIG_FIELD_LABELS = (
    "Model",
    "Camera 0",
    "Camera 1",
    "Width",
    "Height",
    "Camera Target FPS",
    "Exposure",
    "Tracking Threshold",
    "Reject Threshold",
    "Pair Max Skew ms",
    TRIGGER_PIN_LABEL,
    "Trigger Duration",
    "Trigger Min Gap",
    "Timing Camera",
    "Actuation Axis",
    "Actuation Line Ratio",
    "Finalize Quiet ms",
    "Nozzle Distance mm",
    "Belt Speed mm/s",
    "Trigger Offset s",
    "Latency Compensation ms",
    "Preview Lead ms",
    "Timing Log Dir",
    "Debug Dir",
    "Pictures Dir",
)


def create_gui_config() -> RuntimeConfig:
    return RuntimeConfig.defaults()


class ConfigSettingsStore:
    def __init__(self, path: str | os.PathLike[str] = DEFAULT_SETTINGS_PATH):
        self.path = Path(path)

    def load(self) -> RuntimeConfig:
        if not self.path.exists():
            return create_gui_config()
        try:
            return RuntimeConfig.from_json_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return create_gui_config()

    def save(self, config: RuntimeConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(config.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class HistoryRepository:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
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
                    CREATE TABLE IF NOT EXISTS cap_line_history_v3 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        recorded_at TEXT NOT NULL,
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
                    CREATE INDEX IF NOT EXISTS idx_cap_line_history_v3_recorded_at
                    ON cap_line_history_v3 (recorded_at DESC)
                    """
                )

    def insert_record(self, record) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO cap_line_history_v3 (
                        recorded_at, runtime_event_id, result, final_class_name,
                        final_score, decision_source, camera_labels_json,
                        camera_votes_json, anchor_time, trigger_delay_s
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.recorded_at,
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
                SELECT recorded_at, runtime_event_id, result, final_class_name,
                       final_score, decision_source, anchor_time, trigger_delay_s
                FROM cap_line_history_v3
                ORDER BY recorded_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]


class DetectionAppController:
    def __init__(
        self,
        repository: HistoryRepository,
        *,
        detector_runner: Callable[..., None] = run_detection,
        config_factory: Callable[[], RuntimeConfig] = create_gui_config,
    ):
        self.repository = repository
        self.detector_runner = detector_runner
        self.config_factory = config_factory
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
        config = self.config_factory()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._worker_main,
            args=(config,),
            name="cap-line-v3-ui-worker",
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

    def _worker_main(self, config: RuntimeConfig) -> None:
        try:
            callbacks = RuntimeCallbacks(
                preview_callback=self._store_preview,
                history_callback=self._queue_history_record,
                timing_log_callback=self._queue_timing_record,
                performance_callback=self._queue_performance_snapshot,
                log_fn=print,
            )
            self.detector_runner(
                config,
                callbacks,
                stop_event=self.stop_event,
            )
        except Exception as exc:
            traceback.print_exc()
            self._message_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self._message_queue.put(("stopped", None))

    def _store_preview(self, preview_frame) -> None:
        with self._preview_lock:
            self._latest_preview = preview_frame.copy() if hasattr(preview_frame, "copy") else preview_frame

    def _queue_history_record(self, record) -> None:
        self._message_queue.put(("history", record))

    def _queue_timing_record(self, record) -> None:
        self._message_queue.put(("timing", record))

    def _queue_performance_snapshot(self, snapshot) -> None:
        self._message_queue.put(("performance", snapshot))

    def drain_messages(self) -> dict[str, object]:
        history_records = []
        timing_records = []
        latest_performance = None
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
            elif kind == "performance":
                latest_performance = payload
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
            "latest_performance": latest_performance,
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
    def _format_float(value: object, digits: int = 3) -> str:
        if value in (None, ""):
            return "-"
        return f"{float(value):.{digits}f}"


    class DetectionApp(QWidget):
        def __init__(
            self,
            *,
            repository: HistoryRepository | None = None,
            controller: DetectionAppController | None = None,
            settings_store: ConfigSettingsStore | None = None,
        ):
            super().__init__()
            self.repository = repository or HistoryRepository()
            self.settings_store = settings_store or ConfigSettingsStore()
            self.controller = controller or DetectionAppController(self.repository)
            self._loaded_config = self.settings_store.load()
            self.controller.config_factory = self._build_runtime_config
            self.metric_labels: dict[str, QLabel] = {}
            self._closing_after_stop = False
            self.setWindowTitle("Cap Line Inspector V3")
            self.resize(1440, 920)
            self._build_ui()
            self._load_config(self._loaded_config)
            self._load_history_table()
            self._sync_controls()
            self.poll_timer = QTimer(self)
            self.poll_timer.setInterval(LIVE_POLL_INTERVAL_MS)
            self.poll_timer.timeout.connect(self._poll_controller)
            self.poll_timer.start()

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            header = QHBoxLayout()
            title = QLabel("Cap Line Inspector V3")
            title.setStyleSheet("font-size: 20pt; font-weight: 700;")
            header.addWidget(title)
            header.addStretch(1)
            self.status_value = QLabel("Stopped")
            self.status_value.setObjectName("StatusValue")
            header.addWidget(self.status_value)
            root.addLayout(header)
            self.tabs = QTabWidget()
            self.live_tab = QWidget()
            self.config_tab = QWidget()
            self.history_tab = QWidget()
            self.timing_tab = QWidget()
            self.tabs.addTab(self.live_tab, "Live")
            self.tabs.addTab(self.config_tab, "Config")
            self.tabs.addTab(self.history_tab, "History")
            self.tabs.addTab(self.timing_tab, "Timing")
            root.addWidget(self.tabs)
            self._build_live_tab()
            self._build_config_tab()
            self._build_history_tab()
            self._build_timing_tab()

        def _build_live_tab(self) -> None:
            layout = QVBoxLayout(self.live_tab)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            layout.addWidget(splitter, 1)
            preview_group = QGroupBox("Preview")
            preview_layout = QVBoxLayout(preview_group)
            self.preview_label = QLabel("Waiting for preview frames")
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setMinimumSize(760, 420)
            preview_layout.addWidget(self.preview_label)
            splitter.addWidget(preview_group)
            side = QWidget()
            side_layout = QVBoxLayout(side)
            self.start_button = QPushButton("Start")
            self.stop_button = QPushButton("Stop")
            self.start_button.clicked.connect(self._start_detection)
            self.stop_button.clicked.connect(self._stop_detection)
            row = QHBoxLayout()
            row.addWidget(self.start_button)
            row.addWidget(self.stop_button)
            side_layout.addLayout(row)
            metrics = QGroupBox("Session")
            metrics_layout = QGridLayout(metrics)
            for index, key in enumerate(("target_fps", "processed_fps", "preview_fps", "pair_skew", "dropped_pairs", "overlay_age")):
                title = QLabel(key.replace("_", " ").title())
                value = QLabel("-")
                self.metric_labels[key] = value
                metrics_layout.addWidget(title, index, 0)
                metrics_layout.addWidget(value, index, 1)
            side_layout.addWidget(metrics)
            splitter.addWidget(side)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1)

        def _build_config_tab(self) -> None:
            layout = QVBoxLayout(self.config_tab)
            group = QGroupBox("Runtime Config")
            form = QFormLayout(group)
            self.model_input = QLineEdit()
            self.cam0_input = QLineEdit()
            self.cam1_input = QLineEdit()
            self.width_spin = QSpinBox(); self.width_spin.setRange(160, 4096)
            self.height_spin = QSpinBox(); self.height_spin.setRange(120, 4096)
            self.target_fps_spin = QSpinBox(); self.target_fps_spin.setRange(1, 240)
            self.exposure_spin = QSpinBox(); self.exposure_spin.setRange(1, 10000)
            self.tracking_threshold_spin = QDoubleSpinBox(); self.tracking_threshold_spin.setRange(0, 1); self.tracking_threshold_spin.setDecimals(3)
            self.reject_threshold_spin = QDoubleSpinBox(); self.reject_threshold_spin.setRange(0, 1); self.reject_threshold_spin.setDecimals(3)
            self.pair_skew_spin = QDoubleSpinBox(); self.pair_skew_spin.setRange(0, 1000); self.pair_skew_spin.setDecimals(1)
            self.trigger_pin_input = QLineEdit()
            self.trigger_duration_spin = QDoubleSpinBox(); self.trigger_duration_spin.setRange(0.01, 10); self.trigger_duration_spin.setDecimals(3)
            self.trigger_gap_spin = QDoubleSpinBox(); self.trigger_gap_spin.setRange(0, 10); self.trigger_gap_spin.setDecimals(3)
            self.timing_camera_combo = QComboBox(); self.timing_camera_combo.addItems(["0", "1"])
            self.anchor_axis_combo = QComboBox(); self.anchor_axis_combo.addItems(["x", "y"])
            self.anchor_line_spin = QDoubleSpinBox(); self.anchor_line_spin.setRange(0, 1); self.anchor_line_spin.setDecimals(3)
            self.finalize_quiet_spin = QDoubleSpinBox(); self.finalize_quiet_spin.setRange(0, 5000); self.finalize_quiet_spin.setDecimals(1)
            self.nozzle_distance_spin = QDoubleSpinBox(); self.nozzle_distance_spin.setRange(0, 5000); self.nozzle_distance_spin.setDecimals(3)
            self.belt_speed_spin = QDoubleSpinBox(); self.belt_speed_spin.setRange(0.001, 5000); self.belt_speed_spin.setDecimals(3)
            self.trigger_offset_spin = QDoubleSpinBox(); self.trigger_offset_spin.setRange(-5, 5); self.trigger_offset_spin.setDecimals(3)
            self.latency_compensation_spin = QDoubleSpinBox(); self.latency_compensation_spin.setRange(0, 5000); self.latency_compensation_spin.setDecimals(1)
            self.preview_latency_compensation_spin = QDoubleSpinBox(); self.preview_latency_compensation_spin.setRange(0, 5000); self.preview_latency_compensation_spin.setDecimals(1)
            self.timing_log_dir_input = QLineEdit()
            self.debug_dir_input = QLineEdit()
            self.pictures_dir_input = QLineEdit()
            self.simulate_gpio_checkbox = QCheckBox("Simulate GPIO")
            for label, widget in (
                ("Model", self.model_input),
                ("Camera 0", self.cam0_input),
                ("Camera 1", self.cam1_input),
                ("Width", self.width_spin),
                ("Height", self.height_spin),
                ("Camera Target FPS", self.target_fps_spin),
                ("Exposure", self.exposure_spin),
                ("Tracking Threshold", self.tracking_threshold_spin),
                ("Reject Threshold", self.reject_threshold_spin),
                ("Pair Max Skew ms", self.pair_skew_spin),
                (TRIGGER_PIN_LABEL, self.trigger_pin_input),
                ("Trigger Duration", self.trigger_duration_spin),
                ("Trigger Min Gap", self.trigger_gap_spin),
                ("Timing Camera", self.timing_camera_combo),
                ("Actuation Axis", self.anchor_axis_combo),
                ("Actuation Line Ratio", self.anchor_line_spin),
                ("Finalize Quiet ms", self.finalize_quiet_spin),
                ("Nozzle Distance mm", self.nozzle_distance_spin),
                ("Belt Speed mm/s", self.belt_speed_spin),
                ("Trigger Offset s", self.trigger_offset_spin),
                ("Latency Compensation ms", self.latency_compensation_spin),
                ("Preview Lead ms", self.preview_latency_compensation_spin),
                ("Timing Log Dir", self.timing_log_dir_input),
                ("Debug Dir", self.debug_dir_input),
                ("Pictures Dir", self.pictures_dir_input),
            ):
                form.addRow(label, widget)
            form.addRow("", self.simulate_gpio_checkbox)
            layout.addWidget(group)
            layout.addStretch(1)

        def _build_history_tab(self) -> None:
            layout = QVBoxLayout(self.history_tab)
            self.events_table = QTableWidget(0, 6)
            self.events_table.setHorizontalHeaderLabels(["Recorded", "Event", "Result", "Class", "Score", "Source"])
            layout.addWidget(self.events_table)

        def _build_timing_tab(self) -> None:
            layout = QVBoxLayout(self.timing_tab)
            self.timing_table = QTableWidget(0, 8)
            self.timing_table.setHorizontalHeaderLabels(["Recorded", "Event", "Result", "Class", "Anchor", "Requested", "Trigger On", "Late ms"])
            layout.addWidget(self.timing_table)

        def _load_config(self, config: RuntimeConfig) -> None:
            self.model_input.setText(config.model)
            self.cam0_input.setText(config.cameras[0])
            self.cam1_input.setText(config.cameras[1])
            self.width_spin.setValue(config.resolution[0])
            self.height_spin.setValue(config.resolution[1])
            self.target_fps_spin.setValue(config.target_fps)
            self.exposure_spin.setValue(config.exposure)
            self.tracking_threshold_spin.setValue(config.tracking_threshold)
            self.reject_threshold_spin.setValue(config.reject_threshold)
            self.pair_skew_spin.setValue(config.pair_max_skew_ms)
            self.trigger_pin_input.setText(str(config.trigger_pin))
            self.trigger_duration_spin.setValue(config.trigger_duration)
            self.trigger_gap_spin.setValue(config.trigger_min_gap)
            self.timing_camera_combo.setCurrentText(str(config.timing_camera))
            self.anchor_axis_combo.setCurrentText(config.anchor_axis)
            self.anchor_line_spin.setValue(config.anchor_line_ratio)
            self.finalize_quiet_spin.setValue(config.finalize_quiet_ms)
            self.nozzle_distance_spin.setValue(config.nozzle_distance_mm)
            self.belt_speed_spin.setValue(config.belt_speed_mm_per_s)
            self.trigger_offset_spin.setValue(config.trigger_offset_s)
            self.latency_compensation_spin.setValue(config.latency_compensation_ms)
            self.preview_latency_compensation_spin.setValue(config.preview_latency_compensation_ms)
            self.timing_log_dir_input.setText(config.timing_log_dir)
            self.debug_dir_input.setText(config.debug_dir)
            self.pictures_dir_input.setText(config.pictures_dir)
            self.simulate_gpio_checkbox.setChecked(config.simulate_gpio)

        def _build_runtime_config(self) -> RuntimeConfig:
            config = RuntimeConfig(
                model=self.model_input.text().strip() or RuntimeConfig.defaults().model,
                cameras=(self.cam0_input.text().strip() or "0", self.cam1_input.text().strip() or "3"),
                resolution=(self.width_spin.value(), self.height_spin.value()),
                target_fps=self.target_fps_spin.value(),
                exposure=self.exposure_spin.value(),
                tracking_threshold=self.tracking_threshold_spin.value(),
                reject_threshold=self.reject_threshold_spin.value(),
                pair_max_skew_ms=self.pair_skew_spin.value(),
                trigger_pin=self.trigger_pin_input.text().strip() or RuntimeConfig.defaults().trigger_pin,
                trigger_duration=self.trigger_duration_spin.value(),
                trigger_min_gap=self.trigger_gap_spin.value(),
                timing_camera=int(self.timing_camera_combo.currentText()),
                anchor_axis=self.anchor_axis_combo.currentText(),
                anchor_line_ratio=self.anchor_line_spin.value(),
                finalize_quiet_ms=self.finalize_quiet_spin.value(),
                nozzle_distance_mm=self.nozzle_distance_spin.value(),
                belt_speed_mm_per_s=self.belt_speed_spin.value(),
                trigger_offset_s=self.trigger_offset_spin.value(),
                latency_compensation_ms=self.latency_compensation_spin.value(),
                preview_latency_compensation_ms=self.preview_latency_compensation_spin.value(),
                timing_log_dir=self.timing_log_dir_input.text().strip() or RuntimeConfig.defaults().timing_log_dir,
                debug_dir=self.debug_dir_input.text().strip() or RuntimeConfig.defaults().debug_dir,
                pictures_dir=self.pictures_dir_input.text().strip() or RuntimeConfig.defaults().pictures_dir,
                simulate_gpio=self.simulate_gpio_checkbox.isChecked(),
                no_display=True,
            )
            self.settings_store.save(config)
            return config

        def _config_widgets(self):
            return self.config_tab.findChildren(QWidget)

        def _set_config_enabled(self, enabled: bool) -> None:
            for widget in self._config_widgets():
                if widget is not self.config_tab:
                    widget.setEnabled(enabled)

        def _start_detection(self) -> None:
            self.controller.start()
            self._sync_controls()

        def _stop_detection(self) -> None:
            self.controller.stop()
            self._sync_controls()

        def _sync_controls(self) -> None:
            running = self.controller.is_running
            self.start_button.setEnabled(not running)
            self.stop_button.setEnabled(running)
            self.status_value.setText(self.controller.status_text)
            self._set_config_enabled(not running)

        def _poll_controller(self) -> None:
            changes = self.controller.drain_messages()
            if changes["latest_preview"] is not None:
                self._update_preview(changes["latest_preview"])
            if changes["history_records"]:
                self._load_history_table()
            if changes["latest_performance"] is not None:
                self._update_performance(changes["latest_performance"])
            if changes["error"] is not None or changes["stopped"]:
                self._sync_controls()
            if self._closing_after_stop and not self.controller.is_running:
                self.close()

        def _update_preview(self, frame) -> None:
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
            self.preview_label.setPixmap(
                QPixmap.fromImage(image).scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )

        def _update_performance(self, snapshot) -> None:
            self.metric_labels["target_fps"].setText(str(snapshot.target_fps))
            self.metric_labels["processed_fps"].setText(f"{snapshot.processed_fps:.1f}")
            self.metric_labels["preview_fps"].setText(f"{snapshot.preview_fps:.1f}")
            self.metric_labels["pair_skew"].setText("-" if snapshot.latest_pair_skew_ms is None else f"{snapshot.latest_pair_skew_ms:.1f} ms")
            self.metric_labels["dropped_pairs"].setText(str(snapshot.dropped_pairs))
            self.metric_labels["overlay_age"].setText("-" if snapshot.overlay_age_ms is None else f"{snapshot.overlay_age_ms:.1f} ms")

        def _load_history_table(self) -> None:
            rows = self.repository.fetch_events()
            self.events_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                values = [
                    row.get("recorded_at"),
                    row.get("runtime_event_id"),
                    row.get("result"),
                    row.get("final_class_name"),
                    _format_float(row.get("final_score")),
                    row.get("decision_source"),
                ]
                for column, value in enumerate(values):
                    self.events_table.setItem(row_index, column, QTableWidgetItem("" if value is None else str(value)))

        def closeEvent(self, event: QCloseEvent) -> None:
            if self.controller.is_running:
                self._closing_after_stop = True
                self.controller.stop()
                event.ignore()
                return
            event.accept()


    def main() -> None:
        app = QApplication([])
        window = DetectionApp()
        window.show()
        app.exec()

else:
    class DetectionApp:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyQt6 is required to use cap_line_ui_v3.py")


    def main() -> None:
        raise RuntimeError("PyQt6 is required to run cap_line_ui_v3.py")


if __name__ == "__main__":
    main()
