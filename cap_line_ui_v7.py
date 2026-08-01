#!/usr/bin/env python3
"""Standalone PyQt6 operator UI for the v7 cap-inspection runtime.

Slim by design: live dual-camera preview, start/stop, a status bar (GPIO backend,
per-camera FPS / inference ms, run state), session counters, the slim settings
panel, a manual test-fire button (routed through the running runtime's own pin
while detection is active), and a per-cap history table backed by sqlite (every
cap, pass and reject, with its final merged decision).
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable

from cap_line_v7.config import (
    GPIO_BACKENDS,
    RuntimeConfig,
    normalize_gpio_backend,
    normalize_pixel_format,
    validate_config,
)
from cap_line_v7.runtime import resolve_pin_factory, run_detection
from cap_line_v7.types import RuntimeCallbacks


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = str(SCRIPT_DIR / "cap_line_ui_v7_settings.json")
HISTORY_LIMIT = 200
LIVE_POLL_INTERVAL_MS = 16
RUNTIME_LOG_BACKLOG = 500
RUNTIME_LOG_DRAIN_LIMIT = 100
TRIGGER_PIN_LABEL = "Trigger Pin (Jetson physical BOARD number, e.g. 7)"


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
    pin_factory: Callable[..., object] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Fire the solenoid once so an operator can verify the air line.

    Honors ``gpio_backend`` + ``simulate_gpio`` and returns the backend name
    that was used.

    Only for use while detection is stopped: it opens (and on close tears down)
    its own GPIO handle. Jetson.GPIO ``cleanup`` is process-wide per channel, so
    doing this while the runtime holds the same pin would break every later real
    fire. While running, the UI routes test fires through the runtime instead
    (see ``DetectionAppController.request_test_fire``).
    """

    if pin_factory is None or config.simulate_gpio:
        pin_factory = resolve_pin_factory(config)
    pin = pin_factory(config.trigger_pin)
    backend_name = getattr(pin, "backend_name", type(pin).__name__)
    try:
        pin.on()
        sleep_fn(float(config.trigger_duration))
        pin.off()
    finally:
        pin.close()
    return backend_name


class ConfigSettingsStore:
    """Load/save the v7 runtime config to a JSON file."""

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_SETTINGS_PATH):
        self.path = Path(path)
        self.last_load_migrated = False

    def load(self) -> RuntimeConfig:
        self.last_load_migrated = False
        if not self.path.exists():
            return create_gui_config()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise TypeError("settings JSON must contain an object")
            config = RuntimeConfig.from_json_dict(raw)
            # The editor must be able to reopen an electrically unarmed config
            # so the operator can correct its delay. run_detection performs the
            # stricter actuation-readiness validation on Start.
            validate_config(config, require_actuation_ready=False)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return create_gui_config()
        # ``from_json_dict`` performs versioned migrations. Persist the
        # normalized result immediately so an ignored, deployment-local JSON
        # file cannot re-apply an obsolete configuration at every startup.
        if raw != config.to_json_dict():
            try:
                self.save(config)
                self.last_load_migrated = True
            except OSError as exc:
                # The migrated in-memory config is still safer than falling
                # back to the obsolete file; surface persistence failure in
                # the process log for deployments whose UI has no terminal.
                print(f"[SETTINGS][WARN] migrated settings could not be persisted: {exc}")
        return config

    def save(self, config: RuntimeConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config.to_json_dict(), indent=2, sort_keys=True) + "\n"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass


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
        # Keep existing v7 history in place while adding outcome fields needed
        # to distinguish scheduled, executed and failed actuator work.
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cap_line_history_v7 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    result TEXT NOT NULL,
                    class_name TEXT,
                    confidence REAL,
                    cameras_json TEXT NOT NULL,
                    flagged_cameras_json TEXT NOT NULL,
                    requested_fire_time TEXT,
                    actual_fire_time TEXT,
                    fire_suppressed INTEGER NOT NULL DEFAULT 0,
                    inspection_status TEXT NOT NULL DEFAULT 'valid',
                    fire_status TEXT NOT NULL DEFAULT 'not_requested',
                    UNIQUE (event_id, recorded_at)
                )
                """
            )
            columns = {
                str(row[1])
                for row in self._connection.execute("PRAGMA table_info(cap_line_history_v7)").fetchall()
            }
            if "inspection_status" not in columns:
                self._connection.execute(
                    "ALTER TABLE cap_line_history_v7 "
                    "ADD COLUMN inspection_status TEXT NOT NULL DEFAULT 'valid'"
                )
            added_fire_status = "fire_status" not in columns
            if added_fire_status:
                self._connection.execute(
                    "ALTER TABLE cap_line_history_v7 "
                    "ADD COLUMN fire_status TEXT NOT NULL DEFAULT 'not_requested'"
                )
                # Historical rows can prove an execution only when an actual
                # GPIO-on timestamp exists. A requested-only legacy row stays
                # explicitly unknown rather than being presented as fired.
                self._connection.execute(
                    """
                    UPDATE cap_line_history_v7
                    SET fire_status = CASE
                        WHEN actual_fire_time IS NOT NULL THEN 'fired'
                        WHEN fire_suppressed != 0 THEN 'suppressed'
                        WHEN requested_fire_time IS NOT NULL THEN 'legacy_unknown'
                        ELSE 'not_requested'
                    END
                    """
                )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_cap_line_history_v7_recorded_at "
                "ON cap_line_history_v7 (recorded_at DESC)"
            )

    def upsert_record(self, record) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO cap_line_history_v7 (
                    event_id, recorded_at, result, class_name, confidence,
                    cameras_json, flagged_cameras_json, requested_fire_time, actual_fire_time,
                    fire_suppressed, inspection_status, fire_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id, recorded_at) DO UPDATE SET
                    result=excluded.result,
                    class_name=excluded.class_name,
                    confidence=excluded.confidence,
                    cameras_json=excluded.cameras_json,
                    flagged_cameras_json=excluded.flagged_cameras_json,
                    requested_fire_time=excluded.requested_fire_time,
                    actual_fire_time=excluded.actual_fire_time,
                    fire_suppressed=excluded.fire_suppressed,
                    inspection_status=excluded.inspection_status,
                    fire_status=excluded.fire_status
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
                    int(bool(getattr(record, "fire_suppressed", False))),
                    str(getattr(record, "inspection_status", "valid") or "valid"),
                    str(getattr(record, "fire_status", "not_requested") or "not_requested"),
                ),
            )

    def fetch_history(self, limit: int = HISTORY_LIMIT) -> list[dict[str, object]]:
        """Most recent caps first — every cap (pass and reject), one row per cap."""
        rows = self._connection.execute(
            """
            SELECT event_id, recorded_at, result, class_name, confidence,
                   cameras_json, flagged_cameras_json, requested_fire_time, actual_fire_time,
                   fire_suppressed, inspection_status, fire_status
            FROM cap_line_history_v7
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
        self._log_queue: queue.Queue[str] = queue.Queue(maxsize=RUNTIME_LOG_BACKLOG)
        self._preview_lock = threading.Lock()
        self._latest_preview = None
        self._test_fire_requested = threading.Event()

    def start(self) -> bool:
        if self.is_running:
            return False
        config = self.config_factory()
        self.stop_event = threading.Event()
        self._message_queue = queue.Queue()
        self._log_queue = queue.Queue(maxsize=RUNTIME_LOG_BACKLOG)
        self._test_fire_requested.clear()  # never carry a stale request into a new run
        with self._preview_lock:
            self._latest_preview = None
        self.is_running = True
        self.status_text = "Running"
        self._queue_log(
            "[CONFIG] "
            f"schema={getattr(config, 'settings_schema_version', '-')} "
            f"cameras={tuple(config.cameras)} mirrors={tuple(config.mirror_cameras)} "
            f"gate={config.presence_line_axis}@{config.presence_line_ratio:.3f}/"
            f"{config.presence_direction} target_fps={config.target_fps} "
            f"track_timeout_ms={config.track_timeout_ms:.0f} "
            f"classified_frames={getattr(config, 'min_classified_frames', '-')} "
            f"required_cameras={getattr(config, 'required_inspected_cameras', '-')} "
            f"reject_uninspected={getattr(config, 'reject_uninspected', '-')} "
            f"fire_delay_from_gate_s={config.fire_delay_s:.3f}"
        )
        self.worker_thread = threading.Thread(
            target=self._worker_main, args=(config,), name="cap-line-v7-ui-worker", daemon=True
        )
        self.worker_thread.start()
        return True

    def stop(self) -> bool:
        if not self.is_running or self.stop_event is None:
            return False
        self.stop_event.set()
        self.status_text = "Stopping..."
        return True

    def request_test_fire(self) -> bool:
        """Ask the running runtime to pulse its own pin once.

        Returns False when detection is not running (the caller should pulse a
        standalone pin via ``pulse_test_fire`` instead). Routing through the
        runtime avoids opening a second GPIO handle for the same channel, which
        on Jetson.GPIO tears the runtime's pin down when it closes.
        """

        if not self.is_running:
            return False
        self._test_fire_requested.set()
        return True

    def _consume_test_fire_request(self) -> bool:
        if self._test_fire_requested.is_set():
            self._test_fire_requested.clear()
            return True
        return False

    def _worker_main(self, config: RuntimeConfig) -> None:
        try:
            callbacks = RuntimeCallbacks(
                preview_callback=self._store_preview,
                history_callback=lambda record: self._message_queue.put(("history", record)),
                performance_callback=lambda snapshot: self._message_queue.put(("performance", snapshot)),
                log_fn=self._queue_log,
                test_fire_poll=self._consume_test_fire_request,
            )
            self.detector_runner(config, callbacks, self.stop_event)
        except Exception as exc:
            traceback.print_exc()
            self._message_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self._message_queue.put(("stopped", None))

    def _queue_log(self, *values, **kwargs) -> None:
        separator = str(kwargs.get("sep", " "))
        message = separator.join(str(value) for value in values)
        try:
            self._log_queue.put_nowait(message)
        except queue.Full:
            # Keep the newest diagnostics without allowing a repeated camera
            # warning to create an unbounded GUI backlog.
            try:
                self._log_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._log_queue.put_nowait(message)
            except queue.Full:
                pass
        # Retain terminal/journal output for headless service diagnostics while
        # also making the same message visible inside the operator UI.
        try:
            print(message, flush=bool(kwargs.get("flush", False)))
        except Exception:
            pass

    def _store_preview(self, preview_frame) -> None:
        with self._preview_lock:
            self._latest_preview = preview_frame.copy() if hasattr(preview_frame, "copy") else preview_frame

    def drain_messages(self) -> dict[str, object]:
        history_records = []
        latest_performance = None
        latest_error = None
        log_messages = []
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
                log_messages.append(f"[ERROR] {latest_error}")
            elif kind == "stopped":
                stopped = True
                self.is_running = False
                self.worker_thread = None
                self.stop_event = None
                if latest_error is None and not self.status_text.startswith("Error:"):
                    self.status_text = "Stopped"
        for _ in range(RUNTIME_LOG_DRAIN_LIMIT):
            try:
                log_messages.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        with self._preview_lock:
            latest_preview = self._latest_preview
            self._latest_preview = None
        return {
            "history_records": history_records,
            "latest_preview": latest_preview,
            "latest_performance": latest_performance,
            "log_messages": log_messages,
            "error": latest_error,
            "stopped": stopped,
        }


try:
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtGui import QCloseEvent, QColor, QImage, QPixmap
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
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    PYQT_AVAILABLE = True
except ImportError:
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
            self._test_fire_inflight = False
            self.setWindowTitle("Cap Line Inspector v7")
            self.resize(1380, 880)
            self._build_ui()
            if getattr(self.settings_store, "last_load_migrated", False):
                self.runtime_log.appendPlainText(
                    f"[SETTINGS] migrated and saved {getattr(self.settings_store, 'path', 'settings')}"
                )
            self._load_config(self._loaded_config)
            self._load_history_table()
            self._sync_controls()
            self.poll_timer = QTimer(self)
            self.poll_timer.setInterval(LIVE_POLL_INTERVAL_MS)
            self.poll_timer.timeout.connect(self._poll_controller)
            self.poll_timer.start()

        # -- layout ---------------------------------------------------------

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            header = QHBoxLayout()
            title = QLabel("Cap Line Inspector v7")
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
            self.history_tab = QWidget()
            self.tabs.addTab(self.live_tab, "Live")
            self.tabs.addTab(self.config_tab, "Config")
            self.tabs.addTab(self.history_tab, "Cap Log")
            root.addWidget(self.tabs)
            self._build_live_tab()
            self._build_config_tab()
            self._build_history_tab()

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

            # Final per-cap verdict: the merged cross-camera decision the air
            # fire is based on, fed only by per-cap history records.
            verdict_group = QGroupBox("Last Cap Decision")
            verdict_layout = QVBoxLayout(verdict_group)
            self.verdict_label = QLabel("Waiting for the first cap")
            self.verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.verdict_label.setWordWrap(True)
            self.verdict_label.setMinimumHeight(84)
            self._style_verdict(None)
            verdict_layout.addWidget(self.verdict_label)
            side_layout.addWidget(verdict_group)

            counters = QGroupBox("Session")
            counters_layout = QGridLayout(counters)
            for index, key in enumerate(
                (
                    "caps_seen",
                    "rejects",
                    "unknown_inspections",
                    "filtered_tracks",
                    "gpio_backend",
                    "capture_fps",
                    "processed_fps",
                    "throughput_status",
                    "inference_ms",
                    "detected_boxes",
                    "max_detector_confidence",
                    "stale_results",
                    "model_providers",
                )
            ):
                counters_layout.addWidget(QLabel(key.replace("_", " ").title()), index, 0)
                value = QLabel("-")
                value.setWordWrap(True)
                self.metric_labels[key] = value
                counters_layout.addWidget(value, index, 1)
            side_layout.addWidget(counters)

            diagnostics = QGroupBox("Runtime Log")
            diagnostics_layout = QVBoxLayout(diagnostics)
            self.runtime_log = QPlainTextEdit()
            self.runtime_log.setReadOnly(True)
            self.runtime_log.setMinimumHeight(150)
            self.runtime_log.document().setMaximumBlockCount(500)
            diagnostics_layout.addWidget(self.runtime_log)
            side_layout.addWidget(diagnostics, 1)
            side_layout.addStretch(1)
            splitter.addWidget(side)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1)

        def _build_config_tab(self) -> None:
            layout = QVBoxLayout(self.config_tab)
            group = QGroupBox("Runtime Config")
            form = QFormLayout(group)
            self.model_input = QLineEdit()
            self.classifier_model_input = QLineEdit()
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
            self.classifier_imgsz_spin = QSpinBox(); self.classifier_imgsz_spin.setRange(0, 4096)  # 0 = auto
            self.onnx_threads_spin = QSpinBox(); self.onnx_threads_spin.setRange(1, 64)
            self.detect_threshold_spin = QDoubleSpinBox(); self.detect_threshold_spin.setRange(0, 1); self.detect_threshold_spin.setDecimals(3)
            self.frame_dirt_threshold_spin = QDoubleSpinBox(); self.frame_dirt_threshold_spin.setRange(0, 1); self.frame_dirt_threshold_spin.setDecimals(3)
            self.track_dirt_threshold_spin = QDoubleSpinBox(); self.track_dirt_threshold_spin.setRange(0, 1); self.track_dirt_threshold_spin.setDecimals(3)
            self.crop_margin_spin = QDoubleSpinBox(); self.crop_margin_spin.setRange(0, 1); self.crop_margin_spin.setDecimals(3)
            self.classify_band_spin = QDoubleSpinBox(); self.classify_band_spin.setRange(0.05, 1); self.classify_band_spin.setDecimals(3)
            self.max_classified_boxes_spin = QSpinBox(); self.max_classified_boxes_spin.setRange(1, 16)
            self.duplicate_iou_spin = QDoubleSpinBox(); self.duplicate_iou_spin.setRange(0, 1); self.duplicate_iou_spin.setDecimals(3)
            self.track_iou_spin = QDoubleSpinBox(); self.track_iou_spin.setRange(0, 1); self.track_iou_spin.setDecimals(3)
            self.track_timeout_spin = QDoubleSpinBox(); self.track_timeout_spin.setRange(1, 5000); self.track_timeout_spin.setDecimals(1)
            self.min_track_frames_spin = QSpinBox(); self.min_track_frames_spin.setRange(2, 100)
            self.min_track_travel_spin = QDoubleSpinBox(); self.min_track_travel_spin.setRange(0, 20); self.min_track_travel_spin.setDecimals(3)
            self.min_track_directionality_spin = QDoubleSpinBox(); self.min_track_directionality_spin.setRange(0, 1); self.min_track_directionality_spin.setDecimals(3)
            self.min_defect_frames_spin = QSpinBox(); self.min_defect_frames_spin.setRange(2, 100)
            self.min_line_side_frames_spin = QSpinBox(); self.min_line_side_frames_spin.setRange(1, 100)
            self.min_classified_frames_spin = QSpinBox(); self.min_classified_frames_spin.setRange(1, 100)
            self.required_inspected_cameras_spin = QSpinBox(); self.required_inspected_cameras_spin.setRange(1, 2)
            self.reject_uninspected_checkbox = QCheckBox("Reject caps without enough classifier evidence")
            self.presence_line_axis_combo = QComboBox(); self.presence_line_axis_combo.addItems(["x", "y"])
            self.presence_line_ratio_spin = QDoubleSpinBox(); self.presence_line_ratio_spin.setRange(0, 1); self.presence_line_ratio_spin.setDecimals(3)
            self.presence_direction_combo = QComboBox(); self.presence_direction_combo.addItems(["positive", "negative", "either"])
            self.max_track_gap_spin = QDoubleSpinBox(); self.max_track_gap_spin.setRange(1, 10000); self.max_track_gap_spin.setDecimals(1)
            self.presence_clear_spin = QDoubleSpinBox(); self.presence_clear_spin.setRange(0, 10000); self.presence_clear_spin.setDecimals(1)
            self.fire_delay_spin = QDoubleSpinBox(); self.fire_delay_spin.setRange(0, 10); self.fire_delay_spin.setDecimals(3)
            self.merge_window_spin = QDoubleSpinBox(); self.merge_window_spin.setRange(0, 5000); self.merge_window_spin.setDecimals(1)
            self.min_fire_interval_spin = QDoubleSpinBox(); self.min_fire_interval_spin.setRange(0, 5000); self.min_fire_interval_spin.setDecimals(1)
            self.gpio_backend_combo = QComboBox(); self.gpio_backend_combo.addItems(list(GPIO_BACKENDS))
            self.trigger_pin_input = QLineEdit()
            self.trigger_duration_spin = QDoubleSpinBox(); self.trigger_duration_spin.setRange(0.01, 10); self.trigger_duration_spin.setDecimals(3)
            self.trigger_gap_spin = QDoubleSpinBox(); self.trigger_gap_spin.setRange(0, 10); self.trigger_gap_spin.setDecimals(3)
            self.trigger_max_queue_age_spin = QDoubleSpinBox(); self.trigger_max_queue_age_spin.setRange(0, 10000); self.trigger_max_queue_age_spin.setDecimals(1)
            self.trigger_max_lateness_spin = QDoubleSpinBox(); self.trigger_max_lateness_spin.setRange(0, 10000); self.trigger_max_lateness_spin.setDecimals(1)
            self.max_frame_age_spin = QDoubleSpinBox(); self.max_frame_age_spin.setRange(1, 10000); self.max_frame_age_spin.setDecimals(1)
            self.camera_read_timeout_spin = QDoubleSpinBox(); self.camera_read_timeout_spin.setRange(0.1, 60); self.camera_read_timeout_spin.setDecimals(1)
            self.live_preview_fps_spin = QDoubleSpinBox(); self.live_preview_fps_spin.setRange(0, 120); self.live_preview_fps_spin.setDecimals(1)
            self.db_path_input = QLineEdit()
            self.simulate_gpio_checkbox = QCheckBox("Simulate GPIO (no GPIO hardware)")
            for label, widget in (
                ("Cap Detector Model", self.model_input),
                ("Dirt Classifier Model", self.classifier_model_input),
                ("Camera 0", self.cam0_input),
                ("Camera 1", self.cam1_input),
                ("Mirror Camera 0", self.mirror_camera0_checkbox),
                ("Mirror Camera 1", self.mirror_camera1_checkbox),
                ("Width", self.width_spin),
                ("Height", self.height_spin),
                ("Camera Target FPS", self.target_fps_spin),
                ("Exposure", self.exposure_spin),
                ("Pixel Format", self.pixel_format_input),
                ("Detector Input Size (0=auto)", self.imgsz_spin),
                ("Classifier Input Size (0=auto)", self.classifier_imgsz_spin),
                ("ONNX Threads", self.onnx_threads_spin),
                ("Cap Detect Threshold", self.detect_threshold_spin),
                ("Frame Dirt Threshold P(dirt)", self.frame_dirt_threshold_spin),
                ("Track Dirt Threshold (trimmed mean)", self.track_dirt_threshold_spin),
                ("Crop Margin", self.crop_margin_spin),
                ("Classify Band (central frame fraction)", self.classify_band_spin),
                ("Max Classified Boxes / Frame", self.max_classified_boxes_spin),
                ("Duplicate Box IOU", self.duplicate_iou_spin),
                ("Track IOU", self.track_iou_spin),
                ("Track Timeout ms", self.track_timeout_spin),
                ("Minimum Track Frames", self.min_track_frames_spin),
                ("Minimum Track Travel (box widths)", self.min_track_travel_spin),
                ("Minimum Motion Directionality", self.min_track_directionality_spin),
                ("Consecutive Defect Frames", self.min_defect_frames_spin),
                ("Minimum Frames on Each Gate Side", self.min_line_side_frames_spin),
                ("Minimum Classified Frames", self.min_classified_frames_spin),
                ("Required Inspected Cameras", self.required_inspected_cameras_spin),
                ("Fail-Closed Inspection", self.reject_uninspected_checkbox),
                ("Presence Line Axis", self.presence_line_axis_combo),
                ("Presence Gate Ratio", self.presence_line_ratio_spin),
                ("Belt Direction", self.presence_direction_combo),
                ("Maximum Track Observation Gap ms", self.max_track_gap_spin),
                ("Presence Clear ms", self.presence_clear_spin),
                ("Fire Delay from Presence Gate Crossing s", self.fire_delay_spin),
                ("Merge Window ms", self.merge_window_spin),
                ("Min Fire Interval ms", self.min_fire_interval_spin),
                ("GPIO Backend", self.gpio_backend_combo),
                (TRIGGER_PIN_LABEL, self.trigger_pin_input),
                ("Trigger Duration s", self.trigger_duration_spin),
                ("Trigger Min Gap s", self.trigger_gap_spin),
                ("Trigger Max Queue Age ms", self.trigger_max_queue_age_spin),
                ("Trigger Max Lateness ms", self.trigger_max_lateness_spin),
                ("Maximum Processed Frame Age ms", self.max_frame_age_spin),
                ("Camera Read Failure Timeout s", self.camera_read_timeout_spin),
                ("Live Preview FPS", self.live_preview_fps_spin),
                ("History DB Path", self.db_path_input),
            ):
                form.addRow(label, widget)
            form.addRow("", self.simulate_gpio_checkbox)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(group)
            layout.addWidget(scroll)

        def _build_history_tab(self) -> None:
            layout = QVBoxLayout(self.history_tab)
            self.history_table = QTableWidget(0, 10)
            self.history_table.setHorizontalHeaderLabels(
                [
                    "Cap",
                    "Time",
                    "Result",
                    "Inspection",
                    "Class",
                    "Confidence",
                    "Camera(s)",
                    "Air Status",
                    "Requested Fire",
                    "Actual Fire",
                ]
            )
            layout.addWidget(self.history_table)

        # -- config <-> widgets --------------------------------------------

        def _load_config(self, config: RuntimeConfig) -> None:
            self.model_input.setText(config.model)
            self.classifier_model_input.setText(config.classifier_model)
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
            self.classifier_imgsz_spin.setValue(0 if config.classifier_imgsz is None else int(config.classifier_imgsz))
            self.onnx_threads_spin.setValue(config.onnx_intra_op_threads)
            self.detect_threshold_spin.setValue(config.detect_threshold)
            self.frame_dirt_threshold_spin.setValue(config.frame_dirt_threshold)
            self.track_dirt_threshold_spin.setValue(config.track_dirt_threshold)
            self.crop_margin_spin.setValue(config.crop_margin)
            self.classify_band_spin.setValue(config.classify_band_ratio)
            self.max_classified_boxes_spin.setValue(config.max_classified_boxes)
            self.duplicate_iou_spin.setValue(config.duplicate_iou_threshold)
            self.track_iou_spin.setValue(config.track_iou)
            self.track_timeout_spin.setValue(config.track_timeout_ms)
            self.min_track_frames_spin.setValue(config.min_track_frames)
            self.min_track_travel_spin.setValue(config.min_track_travel_ratio)
            self.min_track_directionality_spin.setValue(config.min_track_directionality)
            self.min_defect_frames_spin.setValue(config.min_defect_frames)
            self.min_line_side_frames_spin.setValue(config.min_line_side_frames)
            self.min_classified_frames_spin.setValue(config.min_classified_frames)
            self.required_inspected_cameras_spin.setValue(config.required_inspected_cameras)
            self.reject_uninspected_checkbox.setChecked(config.reject_uninspected)
            self.presence_line_axis_combo.setCurrentText(config.presence_line_axis)
            self.presence_line_ratio_spin.setValue(config.presence_line_ratio)
            self.presence_direction_combo.setCurrentText(config.presence_direction)
            self.max_track_gap_spin.setValue(config.max_track_gap_ms)
            self.presence_clear_spin.setValue(config.presence_clear_ms)
            self.fire_delay_spin.setValue(config.fire_delay_s)
            self.merge_window_spin.setValue(config.merge_window_ms)
            self.min_fire_interval_spin.setValue(config.min_fire_interval_ms)
            self.gpio_backend_combo.setCurrentText(normalize_gpio_backend(config.gpio_backend))
            self.trigger_pin_input.setText(str(config.trigger_pin))
            self.trigger_duration_spin.setValue(config.trigger_duration)
            self.trigger_gap_spin.setValue(config.trigger_min_gap)
            self.trigger_max_queue_age_spin.setValue(config.trigger_max_queue_age_ms)
            self.trigger_max_lateness_spin.setValue(config.trigger_max_lateness_ms)
            self.max_frame_age_spin.setValue(config.max_frame_age_ms)
            self.camera_read_timeout_spin.setValue(config.camera_read_timeout_s)
            self.live_preview_fps_spin.setValue(config.live_preview_fps)
            self.db_path_input.setText(config.db_path)
            self.simulate_gpio_checkbox.setChecked(config.simulate_gpio)

        def _build_runtime_config(self) -> RuntimeConfig:
            defaults = RuntimeConfig.defaults()
            imgsz = self.imgsz_spin.value()
            classifier_imgsz = self.classifier_imgsz_spin.value()
            config = RuntimeConfig(
                settings_schema_version=defaults.settings_schema_version,
                model=self.model_input.text().strip() or defaults.model,
                classifier_model=self.classifier_model_input.text().strip() or defaults.classifier_model,
                cameras=(self.cam0_input.text().strip() or "0", self.cam1_input.text().strip() or "2"),
                mirror_cameras=(self.mirror_camera0_checkbox.isChecked(), self.mirror_camera1_checkbox.isChecked()),
                resolution=(self.width_spin.value(), self.height_spin.value()),
                target_fps=self.target_fps_spin.value(),
                exposure=self.exposure_spin.value(),
                pixel_format=normalize_pixel_format(self.pixel_format_input.text().strip() or defaults.pixel_format),
                imgsz=None if imgsz <= 0 else imgsz,
                classifier_imgsz=None if classifier_imgsz <= 0 else classifier_imgsz,
                onnx_intra_op_threads=self.onnx_threads_spin.value(),
                detect_threshold=self.detect_threshold_spin.value(),
                frame_dirt_threshold=self.frame_dirt_threshold_spin.value(),
                track_dirt_threshold=self.track_dirt_threshold_spin.value(),
                crop_margin=self.crop_margin_spin.value(),
                classify_band_ratio=self.classify_band_spin.value(),
                max_classified_boxes=self.max_classified_boxes_spin.value(),
                duplicate_iou_threshold=self.duplicate_iou_spin.value(),
                track_iou=self.track_iou_spin.value(),
                track_timeout_ms=self.track_timeout_spin.value(),
                min_track_frames=self.min_track_frames_spin.value(),
                min_track_travel_ratio=self.min_track_travel_spin.value(),
                min_track_directionality=self.min_track_directionality_spin.value(),
                min_defect_frames=self.min_defect_frames_spin.value(),
                min_line_side_frames=self.min_line_side_frames_spin.value(),
                min_classified_frames=self.min_classified_frames_spin.value(),
                required_inspected_cameras=self.required_inspected_cameras_spin.value(),
                reject_uninspected=self.reject_uninspected_checkbox.isChecked(),
                presence_line_axis=self.presence_line_axis_combo.currentText(),
                presence_line_ratio=self.presence_line_ratio_spin.value(),
                presence_direction=self.presence_direction_combo.currentText(),
                max_track_gap_ms=self.max_track_gap_spin.value(),
                presence_clear_ms=self.presence_clear_spin.value(),
                fire_delay_s=self.fire_delay_spin.value(),
                merge_window_ms=self.merge_window_spin.value(),
                min_fire_interval_ms=self.min_fire_interval_spin.value(),
                gpio_backend=normalize_gpio_backend(self.gpio_backend_combo.currentText()),
                trigger_pin=self.trigger_pin_input.text().strip() or defaults.trigger_pin,
                trigger_duration=self.trigger_duration_spin.value(),
                trigger_min_gap=self.trigger_gap_spin.value(),
                trigger_max_queue_age_ms=self.trigger_max_queue_age_spin.value(),
                trigger_max_lateness_ms=self.trigger_max_lateness_spin.value(),
                max_frame_age_ms=self.max_frame_age_spin.value(),
                camera_read_timeout_s=self.camera_read_timeout_spin.value(),
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
            # While detection runs, fire through the runtime's own scheduler/pin.
            # A second GPIO handle on the same channel would pump air fine but,
            # on close, Jetson.GPIO cleanup tears down the runtime's pin too —
            # after which real reject fires fail silently.
            if self.controller.request_test_fire():
                print("[TEST FIRE] requested via the running runtime")
                return
            if self._test_fire_inflight:
                return
            config = self._build_runtime_config()
            self._test_fire_inflight = True

            def _worker() -> None:
                try:
                    backend = pulse_test_fire(config)
                    print(f"[TEST FIRE] pulsed via {backend}")
                except Exception as exc:  # noqa: BLE001 - surface but never crash the UI
                    traceback.print_exc()
                    print(f"[TEST FIRE][ERROR] {exc}")
                finally:
                    self._test_fire_inflight = False

            threading.Thread(target=_worker, name="cap-line-v7-test-fire", daemon=True).start()

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

        # -- final verdict banner --------------------------------------------

        def _style_verdict(self, result: str | None) -> None:
            base = "font-size: 15pt; font-weight: 800; border-radius: 6px; padding: 8px;"
            if result == "reject":
                colors = "background-color: #b71c1c; color: white;"
            elif result == "pass":
                colors = "background-color: #1b7a2f; color: white;"
            elif result == "unknown":
                colors = "background-color: #b26a00; color: white;"
            else:
                colors = "background-color: #444444; color: #dddddd;"
            self.verdict_label.setStyleSheet(base + colors)

        def _update_verdict(self, record) -> None:
            cameras = ",".join(str(index) for index in record.flagged_cameras or record.cameras) or "-"
            inspection_status = str(getattr(record, "inspection_status", "valid") or "valid")
            fire_status = str(getattr(record, "fire_status", "not_requested") or "not_requested")
            text = (
                f"CAP {record.event_id}: {record.result.upper()}\n"
                f"{format_prediction_text(record.class_name, record.confidence, digits=2)} (cam {cameras})\n"
                f"inspection={inspection_status}  air={fire_status}"
            )
            self.verdict_label.setText(text)
            self._style_verdict("unknown" if inspection_status == "unknown" and record.result != "reject" else record.result)

        # -- polling --------------------------------------------------------

        def _poll_controller(self) -> None:
            changes = self.controller.drain_messages()
            for message in changes["log_messages"]:
                self.runtime_log.appendPlainText(str(message))
            if changes["latest_preview"] is not None:
                self._update_preview(changes["latest_preview"])
            if changes["history_records"]:
                self._update_verdict(changes["history_records"][-1])
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
            self.metric_labels["caps_seen"].setText(str(snapshot.caps_seen))
            self.metric_labels["rejects"].setText(str(snapshot.rejects))
            self.metric_labels["unknown_inspections"].setText(
                str(getattr(snapshot, "unknown_inspections", 0))
            )
            self.metric_labels["filtered_tracks"].setText(
                str(getattr(snapshot, "filtered_tracks", 0))
            )
            self.metric_labels["gpio_backend"].setText(str(snapshot.gpio_backend))
            self.metric_labels["capture_fps"].setText(_format_tuple(snapshot.capture_fps_by_camera))
            self.metric_labels["processed_fps"].setText(
                _format_tuple(getattr(snapshot, "processed_fps_by_camera", ()))
            )
            throughput_status = str(getattr(snapshot, "throughput_status", "unknown"))
            throughput_detail = str(getattr(snapshot, "throughput_detail", ""))
            self.metric_labels["throughput_status"].setText(
                f"{throughput_status}: {throughput_detail}" if throughput_detail else throughput_status
            )
            self.metric_labels["inference_ms"].setText(_format_tuple(snapshot.inference_ms_by_camera))
            detected_boxes = getattr(snapshot, "detected_boxes_by_camera", ())
            self.metric_labels["detected_boxes"].setText(
                ", ".join(str(int(value)) for value in detected_boxes) if detected_boxes else "-"
            )
            self.metric_labels["max_detector_confidence"].setText(
                _format_tuple(getattr(snapshot, "max_detector_confidence_by_camera", ()), digits=3)
            )
            stale_results = getattr(snapshot, "stale_results_by_camera", ())
            self.metric_labels["stale_results"].setText(
                ", ".join(str(int(value)) for value in stale_results) if stale_results else "-"
            )
            detector_providers = getattr(snapshot, "detector_providers", ())
            classifier_providers = getattr(snapshot, "classifier_providers", ())
            detector_text = ", ".join(str(value) for value in detector_providers) or "-"
            classifier_text = ", ".join(str(value) for value in classifier_providers) or "-"
            self.metric_labels["model_providers"].setText(
                f"detector: {detector_text}\nclassifier: {classifier_text}"
            )

        def _load_history_table(self) -> None:
            rows = self.repository.fetch_history()
            self.history_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                is_reject = row.get("result") == "reject"
                # Rejects show the cameras that flagged the defect; pass caps
                # show every camera that saw the cap. A fail-closed unknown
                # reject has no defect-flagging camera, so show its observers.
                cameras_key = "flagged_cameras_json" if is_reject else "cameras_json"
                try:
                    camera_values = json.loads(row.get(cameras_key) or "[]")
                    if is_reject and not camera_values:
                        camera_values = json.loads(row.get("cameras_json") or "[]")
                    cameras = ", ".join(str(value) for value in camera_values)
                except (TypeError, ValueError):
                    cameras = "-"
                inspection_status = str(row.get("inspection_status") or "valid")
                fire_status = str(row.get("fire_status") or "not_requested")
                values = [
                    row.get("event_id"),
                    row.get("recorded_at"),
                    row.get("result"),
                    inspection_status,
                    row.get("class_name"),
                    _format_float(row.get("confidence")),
                    cameras or "-",
                    fire_status,
                    row.get("requested_fire_time") or "-",
                    # Never substitute the requested target for physical GPIO
                    # activation. A missing actual time means it did not fire.
                    row.get("actual_fire_time") or "-",
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem("" if value is None else str(value))
                    if column == 2:
                        if is_reject:
                            item.setForeground(QColor("#d32f2f"))
                        elif row.get("result") == "unknown":
                            item.setForeground(QColor("#b26a00"))
                        else:
                            item.setForeground(QColor("#2e7d32"))
                    elif column == 7 and fire_status not in {"fired", "not_requested"}:
                        item.setForeground(QColor("#d32f2f"))
                    self.history_table.setItem(row_index, column, item)

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
            raise RuntimeError("PyQt6 is required to use cap_line_ui_v7.py")

    def main() -> None:
        raise RuntimeError("PyQt6 is required to run cap_line_ui_v7.py")


if __name__ == "__main__":
    main()
