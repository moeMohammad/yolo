#!/usr/bin/env python3
"""Standalone PyQt6 operator UI for the v4 cap-inspection runtime.

Slim by design: live dual-camera preview, start/stop, a status bar (GPIO backend,
per-camera FPS / inference ms, run state), session counters, the slim settings
panel, a manual test-fire button, and a recent-rejects table backed by sqlite.
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from gpio_output import GPIOOutputPin

from cap_line_v4.actuation import NullGPIOOutputPin
from cap_line_v4.config import RuntimeConfig, normalize_pixel_format
from cap_line_v4.runtime import run_detection
from cap_line_v4.types import RuntimeCallbacks


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = str(SCRIPT_DIR / "cap_line_ui_v4_settings.json")
REJECT_LIMIT = 100
LIVE_POLL_INTERVAL_MS = 16
TRIGGER_PIN_LABEL = "Trigger GPIO09 (Jetson BOARD pin 7)"


def create_gui_config() -> RuntimeConfig:
    return RuntimeConfig.defaults()


def format_prediction_text(class_name: object, confidence: object, *, digits: int = 3) -> str:
    if class_name in (None, "") or confidence in (None, ""):
        return "-"
    try:
        return f"{class_name} {float(confidence):.{digits}f}"
    except (TypeError, ValueError):
        return str(class_name)


def pulse_test_fire(
    config: RuntimeConfig,
    *,
    pin_factory: Callable[..., object] = GPIOOutputPin,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Fire the solenoid once so an operator can verify the air line.

    Honors ``simulate_gpio`` and returns the backend name that was used.
    """

    pin_cls = NullGPIOOutputPin if config.simulate_gpio else pin_factory
    pin = pin_cls(config.trigger_pin)
    backend_name = getattr(pin, "backend_name", type(pin).__name__)
    try:
        pin.on()
        sleep_fn(float(config.trigger_duration))
        pin.off()
    finally:
        pin.close()
    return backend_name


class ConfigSettingsStore:
    """Load/save the slim v4 config to a JSON file."""

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
            json.dumps(config.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class HistoryRepository:
    """One row per physical cap, upserted by ``event_id``.

    Backed by a single connection. It is only ever touched from the UI poll
    thread (the runtime worker queues records; the UI drains and writes them), so
    a persistent ``check_same_thread=False`` connection is both safe and simple,
    and it keeps an in-memory test DB alive across calls.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cap_line_history_v4 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL UNIQUE,
                    recorded_at TEXT NOT NULL,
                    result TEXT NOT NULL,
                    class_name TEXT,
                    confidence REAL,
                    cameras_json TEXT NOT NULL,
                    flagged_cameras_json TEXT NOT NULL,
                    requested_fire_time TEXT,
                    actual_fire_time TEXT
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_cap_line_history_v4_recorded_at "
                "ON cap_line_history_v4 (recorded_at DESC)"
            )

    def upsert_record(self, record) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO cap_line_history_v4 (
                    event_id, recorded_at, result, class_name, confidence,
                    cameras_json, flagged_cameras_json, requested_fire_time, actual_fire_time
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    recorded_at=excluded.recorded_at,
                    result=excluded.result,
                    class_name=excluded.class_name,
                    confidence=excluded.confidence,
                    cameras_json=excluded.cameras_json,
                    flagged_cameras_json=excluded.flagged_cameras_json,
                    requested_fire_time=excluded.requested_fire_time,
                    actual_fire_time=excluded.actual_fire_time
                """,
                (
                    int(record.event_id),
                    record.recorded_at,
                    record.result,
                    record.class_name,
                    record.confidence,
                    json.dumps(list(record.cameras)),
                    json.dumps(list(record.flagged_cameras)),
                    record.requested_fire_time,
                    record.actual_fire_time,
                ),
            )

    def fetch_rejects(self, limit: int = REJECT_LIMIT) -> list[dict[str, object]]:
        rows = self._connection.execute(
            """
            SELECT event_id, recorded_at, result, class_name, confidence,
                   cameras_json, flagged_cameras_json, requested_fire_time, actual_fire_time
            FROM cap_line_history_v4
            WHERE result = 'reject'
            ORDER BY recorded_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(row) for row in rows]


class DetectionAppController:
    """Owns the worker thread that runs ``run_detection`` and a message queue."""

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
        self._message_queue = queue.Queue()
        with self._preview_lock:
            self._latest_preview = None
        self.is_running = True
        self.status_text = "Running"
        self.worker_thread = threading.Thread(
            target=self._worker_main, args=(config,), name="cap-line-v4-ui-worker", daemon=True
        )
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
                history_callback=lambda record: self._message_queue.put(("history", record)),
                performance_callback=lambda snapshot: self._message_queue.put(("performance", snapshot)),
                log_fn=print,
            )
            self.detector_runner(config, callbacks, self.stop_event)
        except Exception as exc:
            traceback.print_exc()
            self._message_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self._message_queue.put(("stopped", None))

    def _store_preview(self, preview_frame) -> None:
        with self._preview_lock:
            self._latest_preview = preview_frame.copy() if hasattr(preview_frame, "copy") else preview_frame

    def drain_messages(self) -> dict[str, object]:
        history_records = []
        latest_performance = None
        latest_error = None
        stopped = False
        while True:
            try:
                kind, payload = self._message_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "history":
                self.repository.upsert_record(payload)
                history_records.append(payload)
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
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
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

    def _format_tuple(values, digits: int = 1) -> str:
        if not values:
            return "-"
        return ", ".join("-" if value is None else f"{float(value):.{digits}f}" for value in values)

    class DetectionApp(QWidget):
        def __init__(
            self,
            *,
            repository: HistoryRepository | None = None,
            controller: DetectionAppController | None = None,
            settings_store: ConfigSettingsStore | None = None,
        ):
            super().__init__()
            self.settings_store = settings_store or ConfigSettingsStore()
            self._loaded_config = self.settings_store.load()
            self.repository = repository or HistoryRepository(self._loaded_config.db_path)
            self.controller = controller or DetectionAppController(self.repository)
            self.controller.config_factory = self._build_runtime_config
            self.metric_labels: dict[str, QLabel] = {}
            self._closing_after_stop = False
            self.setWindowTitle("Cap Line Inspector v4")
            self.resize(1380, 880)
            self._build_ui()
            self._load_config(self._loaded_config)
            self._load_rejects_table()
            self._sync_controls()
            self.poll_timer = QTimer(self)
            self.poll_timer.setInterval(LIVE_POLL_INTERVAL_MS)
            self.poll_timer.timeout.connect(self._poll_controller)
            self.poll_timer.start()

        # -- layout ---------------------------------------------------------

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            header = QHBoxLayout()
            title = QLabel("Cap Line Inspector v4")
            title.setStyleSheet("font-size: 20pt; font-weight: 700;")
            header.addWidget(title)
            header.addStretch(1)
            self.status_value = QLabel("Stopped")
            self.status_value.setStyleSheet("font-size: 13pt; font-weight: 600;")
            header.addWidget(self.status_value)
            root.addLayout(header)
            self.tabs = QTabWidget()
            self.live_tab = QWidget()
            self.config_tab = QWidget()
            self.rejects_tab = QWidget()
            self.tabs.addTab(self.live_tab, "Live")
            self.tabs.addTab(self.config_tab, "Config")
            self.tabs.addTab(self.rejects_tab, "Rejects")
            root.addWidget(self.tabs)
            self._build_live_tab()
            self._build_config_tab()
            self._build_rejects_tab()

        def _build_live_tab(self) -> None:
            layout = QVBoxLayout(self.live_tab)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            layout.addWidget(splitter, 1)
            preview_group = QGroupBox("Dual-Camera Preview")
            preview_layout = QVBoxLayout(preview_group)
            self.preview_label = QLabel("Waiting for preview frames")
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setMinimumSize(760, 420)
            preview_layout.addWidget(self.preview_label)
            splitter.addWidget(preview_group)

            side = QWidget()
            side_layout = QVBoxLayout(side)
            row = QHBoxLayout()
            self.start_button = QPushButton("Start")
            self.stop_button = QPushButton("Stop")
            self.start_button.clicked.connect(self._start_detection)
            self.stop_button.clicked.connect(self._stop_detection)
            row.addWidget(self.start_button)
            row.addWidget(self.stop_button)
            side_layout.addLayout(row)

            self.test_fire_button = QPushButton("Test Fire")
            self.test_fire_button.clicked.connect(self._test_fire)
            side_layout.addWidget(self.test_fire_button)

            counters = QGroupBox("Session")
            counters_layout = QGridLayout(counters)
            for index, key in enumerate(
                ("caps_seen", "rejects", "gpio_backend", "capture_fps", "inference_ms")
            ):
                counters_layout.addWidget(QLabel(key.replace("_", " ").title()), index, 0)
                value = QLabel("-")
                self.metric_labels[key] = value
                counters_layout.addWidget(value, index, 1)
            side_layout.addWidget(counters)
            side_layout.addStretch(1)
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
            self.mirror_camera0_checkbox = QCheckBox()
            self.mirror_camera1_checkbox = QCheckBox()
            self.width_spin = QSpinBox(); self.width_spin.setRange(160, 4096)
            self.height_spin = QSpinBox(); self.height_spin.setRange(120, 4096)
            self.target_fps_spin = QSpinBox(); self.target_fps_spin.setRange(1, 240)
            self.exposure_spin = QSpinBox(); self.exposure_spin.setRange(1, 10000)
            self.pixel_format_input = QLineEdit()
            self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(0, 4096)  # 0 = auto
            self.onnx_threads_spin = QSpinBox(); self.onnx_threads_spin.setRange(1, 64)
            self.reject_threshold_spin = QDoubleSpinBox(); self.reject_threshold_spin.setRange(0, 1); self.reject_threshold_spin.setDecimals(3)
            self.track_iou_spin = QDoubleSpinBox(); self.track_iou_spin.setRange(0, 1); self.track_iou_spin.setDecimals(3)
            self.track_timeout_spin = QDoubleSpinBox(); self.track_timeout_spin.setRange(0, 5000); self.track_timeout_spin.setDecimals(1)
            self.fire_delay_spin = QDoubleSpinBox(); self.fire_delay_spin.setRange(0, 10); self.fire_delay_spin.setDecimals(3)
            self.global_cooldown_spin = QDoubleSpinBox(); self.global_cooldown_spin.setRange(0, 5000); self.global_cooldown_spin.setDecimals(1)
            self.trigger_pin_input = QLineEdit()
            self.trigger_duration_spin = QDoubleSpinBox(); self.trigger_duration_spin.setRange(0.01, 10); self.trigger_duration_spin.setDecimals(3)
            self.trigger_gap_spin = QDoubleSpinBox(); self.trigger_gap_spin.setRange(0, 10); self.trigger_gap_spin.setDecimals(3)
            self.live_preview_fps_spin = QDoubleSpinBox(); self.live_preview_fps_spin.setRange(0, 120); self.live_preview_fps_spin.setDecimals(1)
            self.db_path_input = QLineEdit()
            self.simulate_gpio_checkbox = QCheckBox("Simulate GPIO (no Jetson hardware)")
            for label, widget in (
                ("Model", self.model_input),
                ("Camera 0", self.cam0_input),
                ("Camera 1", self.cam1_input),
                ("Mirror Camera 0", self.mirror_camera0_checkbox),
                ("Mirror Camera 1", self.mirror_camera1_checkbox),
                ("Width", self.width_spin),
                ("Height", self.height_spin),
                ("Camera Target FPS", self.target_fps_spin),
                ("Exposure", self.exposure_spin),
                ("Pixel Format", self.pixel_format_input),
                ("Model Input Size (0=auto)", self.imgsz_spin),
                ("ONNX Threads", self.onnx_threads_spin),
                ("Reject Threshold", self.reject_threshold_spin),
                ("Track IOU", self.track_iou_spin),
                ("Track Timeout ms", self.track_timeout_spin),
                ("Fire Delay s", self.fire_delay_spin),
                ("Global Cooldown ms", self.global_cooldown_spin),
                (TRIGGER_PIN_LABEL, self.trigger_pin_input),
                ("Trigger Duration s", self.trigger_duration_spin),
                ("Trigger Min Gap s", self.trigger_gap_spin),
                ("Live Preview FPS", self.live_preview_fps_spin),
                ("History DB Path", self.db_path_input),
            ):
                form.addRow(label, widget)
            form.addRow("", self.simulate_gpio_checkbox)
            layout.addWidget(group)
            layout.addStretch(1)

        def _build_rejects_tab(self) -> None:
            layout = QVBoxLayout(self.rejects_tab)
            self.rejects_table = QTableWidget(0, 6)
            self.rejects_table.setHorizontalHeaderLabels(
                ["Time", "Result", "Class", "Confidence", "Camera(s)", "Fire Time"]
            )
            layout.addWidget(self.rejects_table)

        # -- config <-> widgets --------------------------------------------

        def _load_config(self, config: RuntimeConfig) -> None:
            self.model_input.setText(config.model)
            self.cam0_input.setText(config.cameras[0])
            self.cam1_input.setText(config.cameras[1])
            self.mirror_camera0_checkbox.setChecked(config.mirror_cameras[0])
            self.mirror_camera1_checkbox.setChecked(config.mirror_cameras[1])
            self.width_spin.setValue(config.resolution[0])
            self.height_spin.setValue(config.resolution[1])
            self.target_fps_spin.setValue(config.target_fps)
            self.exposure_spin.setValue(config.exposure)
            self.pixel_format_input.setText(config.pixel_format)
            self.imgsz_spin.setValue(0 if config.imgsz is None else int(config.imgsz))
            self.onnx_threads_spin.setValue(config.onnx_intra_op_threads)
            self.reject_threshold_spin.setValue(config.reject_threshold)
            self.track_iou_spin.setValue(config.track_iou)
            self.track_timeout_spin.setValue(config.track_timeout_ms)
            self.fire_delay_spin.setValue(config.fire_delay_s)
            self.global_cooldown_spin.setValue(config.global_cooldown_ms)
            self.trigger_pin_input.setText(str(config.trigger_pin))
            self.trigger_duration_spin.setValue(config.trigger_duration)
            self.trigger_gap_spin.setValue(config.trigger_min_gap)
            self.live_preview_fps_spin.setValue(config.live_preview_fps)
            self.db_path_input.setText(config.db_path)
            self.simulate_gpio_checkbox.setChecked(config.simulate_gpio)

        def _build_runtime_config(self) -> RuntimeConfig:
            defaults = RuntimeConfig.defaults()
            imgsz = self.imgsz_spin.value()
            config = RuntimeConfig(
                model=self.model_input.text().strip() or defaults.model,
                cameras=(self.cam0_input.text().strip() or "0", self.cam1_input.text().strip() or "3"),
                mirror_cameras=(self.mirror_camera0_checkbox.isChecked(), self.mirror_camera1_checkbox.isChecked()),
                resolution=(self.width_spin.value(), self.height_spin.value()),
                target_fps=self.target_fps_spin.value(),
                exposure=self.exposure_spin.value(),
                pixel_format=normalize_pixel_format(self.pixel_format_input.text().strip() or defaults.pixel_format),
                imgsz=None if imgsz <= 0 else imgsz,
                onnx_intra_op_threads=self.onnx_threads_spin.value(),
                reject_threshold=self.reject_threshold_spin.value(),
                track_iou=self.track_iou_spin.value(),
                track_timeout_ms=self.track_timeout_spin.value(),
                fire_delay_s=self.fire_delay_spin.value(),
                global_cooldown_ms=self.global_cooldown_spin.value(),
                trigger_pin=self.trigger_pin_input.text().strip() or defaults.trigger_pin,
                trigger_duration=self.trigger_duration_spin.value(),
                trigger_min_gap=self.trigger_gap_spin.value(),
                live_preview_fps=self.live_preview_fps_spin.value(),
                db_path=self.db_path_input.text().strip() or defaults.db_path,
                simulate_gpio=self.simulate_gpio_checkbox.isChecked(),
                no_display=False,
            )
            self.settings_store.save(config)
            return config

        # -- actions --------------------------------------------------------

        def _start_detection(self) -> None:
            self.controller.start()
            self._sync_controls()

        def _stop_detection(self) -> None:
            self.controller.stop()
            self._sync_controls()

        def _test_fire(self) -> None:
            config = self._build_runtime_config()

            def _worker() -> None:
                try:
                    backend = pulse_test_fire(config)
                    print(f"[TEST FIRE] pulsed via {backend}")
                except Exception as exc:  # noqa: BLE001 - surface but never crash the UI
                    traceback.print_exc()
                    print(f"[TEST FIRE][ERROR] {exc}")

            threading.Thread(target=_worker, name="cap-line-v4-test-fire", daemon=True).start()

        def _config_widgets(self):
            return self.config_tab.findChildren(QWidget)

        def _set_config_enabled(self, enabled: bool) -> None:
            for widget in self._config_widgets():
                if widget is not self.config_tab:
                    widget.setEnabled(enabled)

        def _sync_controls(self) -> None:
            running = self.controller.is_running
            self.start_button.setEnabled(not running)
            self.stop_button.setEnabled(running)
            self.status_value.setText(self.controller.status_text)
            self._set_config_enabled(not running)

        # -- polling --------------------------------------------------------

        def _poll_controller(self) -> None:
            changes = self.controller.drain_messages()
            if changes["latest_preview"] is not None:
                self._update_preview(changes["latest_preview"])
            if changes["history_records"]:
                self._load_rejects_table()
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
            self.metric_labels["caps_seen"].setText(str(snapshot.caps_seen))
            self.metric_labels["rejects"].setText(str(snapshot.rejects))
            self.metric_labels["gpio_backend"].setText(str(snapshot.gpio_backend))
            self.metric_labels["capture_fps"].setText(_format_tuple(snapshot.capture_fps_by_camera))
            self.metric_labels["inference_ms"].setText(_format_tuple(snapshot.inference_ms_by_camera))

        def _load_rejects_table(self) -> None:
            rows = self.repository.fetch_rejects()
            self.rejects_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                try:
                    cameras = ", ".join(str(value) for value in json.loads(row.get("flagged_cameras_json") or "[]"))
                except (TypeError, ValueError):
                    cameras = "-"
                values = [
                    row.get("recorded_at"),
                    row.get("result"),
                    row.get("class_name"),
                    _format_float(row.get("confidence")),
                    cameras or "-",
                    row.get("actual_fire_time") or row.get("requested_fire_time") or "-",
                ]
                for column, value in enumerate(values):
                    self.rejects_table.setItem(row_index, column, QTableWidgetItem("" if value is None else str(value)))

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
            raise RuntimeError("PyQt6 is required to use cap_line_ui_v4.py")

    def main() -> None:
        raise RuntimeError("PyQt6 is required to run cap_line_ui_v4.py")


if __name__ == "__main__":
    main()
