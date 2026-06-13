#!/usr/bin/env python3
"""
PyQt operator UI for the greyscale V2 cap inspection runtime.
"""

from __future__ import annotations

import os
from typing import Callable

import cap_line_ui as base_ui
import cap_line_runtime_v2_grey as cap_line_runtime_v2
from cap_line_runtime_v2_grey import DetectionHistoryRecord, TimingLogRecord


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "data", "cap_line_history_v2.sqlite3")
DEFAULT_MODEL = cap_line_runtime_v2.DEFAULT_MODEL
DEFAULT_TIMING_LOG_DIR = cap_line_runtime_v2.DEFAULT_TIMING_LOG_DIR
DEFAULT_DEBUG_DIR = cap_line_runtime_v2.DEFAULT_DEBUG_DIR
DEFAULT_PICTURES_DIR = cap_line_runtime_v2.DEFAULT_PICTURES_DIR
EVENT_LIMIT = base_ui.EVENT_LIMIT
TIMING_LOG_LIMIT = base_ui.TIMING_LOG_LIMIT
DETECTION_RULE_LABEL = "Tracking and reject thresholds are configurable"
CALIBRATION_FIELD_LABELS = (
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
    base_ui.TRIGGER_PIN_LABEL,
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
    "Timing Log Dir",
    "Debug Dir",
    "Pictures Dir",
)
PYQT_AVAILABLE = base_ui.PYQT_AVAILABLE
read_recent_timing_logs = base_ui.read_recent_timing_logs


class HistoryRepository(base_ui.HistoryRepository):
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        super().__init__(db_path=db_path)


def create_gui_args():
    args = cap_line_runtime_v2.parse_args([])
    args.model = DEFAULT_MODEL
    args.no_display = True
    args.timing_log_dir = DEFAULT_TIMING_LOG_DIR
    args.debug_dir = DEFAULT_DEBUG_DIR
    args.pictures_dir = DEFAULT_PICTURES_DIR
    args.review_dir = DEFAULT_DEBUG_DIR
    args.session_log_dir = cap_line_runtime_v2.DEFAULT_SESSION_LOG_DIR
    args.global_threshold = cap_line_runtime_v2.GLOBAL_DETECTION_THRESHOLD
    args.tracking_threshold = cap_line_runtime_v2.TRACKING_DETECTION_THRESHOLD
    args.reject_threshold = cap_line_runtime_v2.DEFECT_REJECT_THRESHOLD
    args.pair_max_skew_ms = cap_line_runtime_v2.DEFAULT_PAIR_MAX_SKEW_MS
    args.save_queue_warning_threshold = (
        cap_line_runtime_v2.DEFAULT_SAVE_QUEUE_WARNING_THRESHOLD
    )
    if os.name == "nt":
        args.simulate_gpio = True
    return args


class DetectionAppController(base_ui.DetectionAppController):
    def __init__(
        self,
        repository: HistoryRepository,
        *,
        detector_runner: Callable[..., None] = cap_line_runtime_v2.run_detection,
        args_factory: Callable[[], object] = create_gui_args,
    ):
        super().__init__(
            repository,
            detector_runner=detector_runner,
            args_factory=args_factory,
        )

    def _queue_performance_snapshot(self, snapshot) -> None:
        self._message_queue.put(("performance", snapshot))

    def _worker_main(self, args) -> None:
        try:
            self.detector_runner(
                args,
                stop_event=self.stop_event,
                preview_callback=self._store_preview,
                history_callback=self._queue_history_record,
                timing_log_callback=self._queue_timing_record,
                performance_callback=self._queue_performance_snapshot,
                log_fn=print,
            )
        except Exception as exc:
            base_ui.traceback.print_exc()
            self._message_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self._message_queue.put(("stopped", None))

    def drain_messages(self) -> dict[str, object]:
        history_records: list[DetectionHistoryRecord] = []
        timing_records: list[TimingLogRecord] = []
        latest_performance = None
        latest_error = None
        stopped = False

        while True:
            try:
                kind, payload = self._message_queue.get_nowait()
            except base_ui.queue.Empty:
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


if PYQT_AVAILABLE:
    class DetectionApp(base_ui.DetectionApp):
        def __init__(
            self,
            *,
            repository: HistoryRepository | None = None,
            controller: DetectionAppController | None = None,
        ):
            repository = repository or HistoryRepository()
            controller = controller or DetectionAppController(repository)
            super().__init__(repository=repository, controller=controller)
            self.setWindowTitle("Cap Line Inspector V2 Grey")

        def _build_live_tab(self) -> None:
            layout = base_ui.QVBoxLayout(self.live_tab)
            layout.setSpacing(12)

            top_splitter = base_ui.QSplitter(base_ui.Qt.Orientation.Horizontal)
            layout.addWidget(top_splitter, 1)

            preview_group = base_ui.QGroupBox("Preview")
            preview_layout = base_ui.QVBoxLayout(preview_group)
            self.preview_label = base_ui.QLabel("Waiting for preview frames")
            self.preview_label.setObjectName("PreviewSurface")
            self.preview_label.setAlignment(base_ui.Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setMinimumSize(760, 420)
            preview_layout.addWidget(self.preview_label)
            top_splitter.addWidget(preview_group)

            side_panel = base_ui.QWidget()
            side_layout = base_ui.QVBoxLayout(side_panel)
            side_layout.setSpacing(12)
            top_splitter.addWidget(side_panel)
            top_splitter.setStretchFactor(0, 3)
            top_splitter.setStretchFactor(1, 2)

            control_group = base_ui.QGroupBox("Controls")
            control_layout = base_ui.QVBoxLayout(control_group)
            button_row = base_ui.QHBoxLayout()
            self.start_button = base_ui.QPushButton("Start")
            self.stop_button = base_ui.QPushButton("Stop")
            self.start_button.clicked.connect(self._start_detection)
            self.stop_button.clicked.connect(self._stop_detection)
            button_row.addWidget(self.start_button)
            button_row.addWidget(self.stop_button)
            control_layout.addLayout(button_row)
            side_layout.addWidget(control_group)

            metrics_group = base_ui.QGroupBox("Session")
            metrics_layout = base_ui.QGridLayout(metrics_group)
            self.metric_labels: dict[str, base_ui.QLabel] = {}
            metric_defs = [
                ("Total Items", "total_items"),
                ("Reject Triggers", "reject_triggers"),
                ("Undefected Wins", "undefected_wins"),
                ("Dirt Defects", "dirt_defects"),
                ("Reject Rate", "reject_rate"),
                ("Pair Skew", "pair_skew"),
                ("Dropped Pairs", "dropped_pairs"),
            ]
            for index, (label_text, key) in enumerate(metric_defs):
                title = base_ui.QLabel(label_text)
                title.setObjectName("MetricTitle")
                value = base_ui.QLabel("0")
                value.setObjectName("MetricValue")
                self.metric_labels[key] = value
                metrics_layout.addWidget(title, index, 0)
                metrics_layout.addWidget(value, index, 1)
            side_layout.addWidget(metrics_group)

            config_group = base_ui.QGroupBox("Calibration")
            config_form = base_ui.QFormLayout(config_group)
            self.model_input = base_ui.QLineEdit(DEFAULT_MODEL)
            self.cam0_input = base_ui.QLineEdit("0")
            self.cam1_input = base_ui.QLineEdit("1")
            self.width_spin = base_ui.QSpinBox()
            self.width_spin.setRange(160, 4096)
            self.width_spin.setValue(cap_line_runtime_v2.DEFAULT_CAMERA_RESOLUTION[0])
            self.height_spin = base_ui.QSpinBox()
            self.height_spin.setRange(120, 4096)
            self.height_spin.setValue(cap_line_runtime_v2.DEFAULT_CAMERA_RESOLUTION[1])
            self.fps_spin = base_ui.QSpinBox()
            self.fps_spin.setRange(1, 240)
            self.fps_spin.setValue(cap_line_runtime_v2.DEFAULT_CAMERA_FPS)
            self.exposure_spin = base_ui.QSpinBox()
            self.exposure_spin.setRange(1, 10000)
            self.exposure_spin.setValue(8)
            self.tracking_threshold_spin = base_ui.QDoubleSpinBox()
            self.tracking_threshold_spin.setRange(0.0, 1.0)
            self.tracking_threshold_spin.setDecimals(3)
            self.tracking_threshold_spin.setSingleStep(0.01)
            self.tracking_threshold_spin.setValue(
                cap_line_runtime_v2.TRACKING_DETECTION_THRESHOLD
            )
            self.reject_threshold_spin = base_ui.QDoubleSpinBox()
            self.reject_threshold_spin.setRange(0.0, 1.0)
            self.reject_threshold_spin.setDecimals(3)
            self.reject_threshold_spin.setSingleStep(0.01)
            self.reject_threshold_spin.setValue(
                cap_line_runtime_v2.DEFECT_REJECT_THRESHOLD
            )
            self.pair_skew_spin = base_ui.QDoubleSpinBox()
            self.pair_skew_spin.setRange(0.0, 1000.0)
            self.pair_skew_spin.setDecimals(1)
            self.pair_skew_spin.setSingleStep(5.0)
            self.pair_skew_spin.setValue(cap_line_runtime_v2.DEFAULT_PAIR_MAX_SKEW_MS)
            self.trigger_pin_input = base_ui.QLineEdit(
                cap_line_runtime_v2.DEFAULT_TRIGGER_PIN
            )
            self.trigger_duration_spin = base_ui.QDoubleSpinBox()
            self.trigger_duration_spin.setRange(0.01, 10.0)
            self.trigger_duration_spin.setDecimals(3)
            self.trigger_duration_spin.setValue(0.3)
            self.trigger_gap_spin = base_ui.QDoubleSpinBox()
            self.trigger_gap_spin.setRange(0.0, 10.0)
            self.trigger_gap_spin.setDecimals(3)
            self.trigger_gap_spin.setValue(0.0)
            self.timing_camera_combo = base_ui.QComboBox()
            self.timing_camera_combo.addItems(["0", "1"])
            self.anchor_axis_combo = base_ui.QComboBox()
            self.anchor_axis_combo.addItems(["x", "y"])
            self.anchor_line_spin = base_ui.QDoubleSpinBox()
            self.anchor_line_spin.setRange(0.0, 1.0)
            self.anchor_line_spin.setDecimals(3)
            self.anchor_line_spin.setSingleStep(0.05)
            self.anchor_line_spin.setValue(0.5)
            self.finalize_quiet_spin = base_ui.QDoubleSpinBox()
            self.finalize_quiet_spin.setRange(0.0, 5000.0)
            self.finalize_quiet_spin.setDecimals(1)
            self.finalize_quiet_spin.setSingleStep(5.0)
            self.finalize_quiet_spin.setValue(cap_line_runtime_v2.DEFAULT_FINALIZE_QUIET_MS)
            self.nozzle_distance_spin = base_ui.QDoubleSpinBox()
            self.nozzle_distance_spin.setRange(0.0, 5000.0)
            self.nozzle_distance_spin.setDecimals(3)
            self.nozzle_distance_spin.setValue(cap_line_runtime_v2.DEFAULT_NOZZLE_DISTANCE_MM)
            self.belt_speed_spin = base_ui.QDoubleSpinBox()
            self.belt_speed_spin.setRange(0.001, 5000.0)
            self.belt_speed_spin.setDecimals(3)
            self.belt_speed_spin.setValue(cap_line_runtime_v2.DEFAULT_BELT_SPEED_MM_PER_S)
            self.trigger_offset_spin = base_ui.QDoubleSpinBox()
            self.trigger_offset_spin.setRange(-5.0, 5.0)
            self.trigger_offset_spin.setDecimals(3)
            self.trigger_offset_spin.setValue(cap_line_runtime_v2.DEFAULT_TRIGGER_OFFSET_S)
            self.latency_compensation_spin = base_ui.QDoubleSpinBox()
            self.latency_compensation_spin.setRange(0.0, 5000.0)
            self.latency_compensation_spin.setDecimals(1)
            self.latency_compensation_spin.setSingleStep(5.0)
            self.latency_compensation_spin.setValue(
                cap_line_runtime_v2.DEFAULT_LATENCY_COMPENSATION_MS
            )
            self.timing_log_dir_input = base_ui.QLineEdit(DEFAULT_TIMING_LOG_DIR)
            self.debug_dir_input = base_ui.QLineEdit(DEFAULT_DEBUG_DIR)
            self.pictures_dir_input = base_ui.QLineEdit(DEFAULT_PICTURES_DIR)
            self.simulate_gpio_checkbox = base_ui.QCheckBox("Simulate GPIO")
            self.simulate_gpio_checkbox.setChecked(os.name == "nt")

            config_form.addRow("Model", self.model_input)
            config_form.addRow("Camera 0", self.cam0_input)
            config_form.addRow("Camera 1", self.cam1_input)
            config_form.addRow("Width", self.width_spin)
            config_form.addRow("Height", self.height_spin)
            config_form.addRow("Camera Target FPS", self.fps_spin)
            config_form.addRow("Exposure", self.exposure_spin)
            config_form.addRow("Tracking Threshold", self.tracking_threshold_spin)
            config_form.addRow("Reject Threshold", self.reject_threshold_spin)
            config_form.addRow("Pair Max Skew ms", self.pair_skew_spin)
            config_form.addRow(base_ui.TRIGGER_PIN_LABEL, self.trigger_pin_input)
            config_form.addRow("Trigger Duration", self.trigger_duration_spin)
            config_form.addRow("Trigger Min Gap", self.trigger_gap_spin)
            config_form.addRow("Timing Camera", self.timing_camera_combo)
            config_form.addRow("Actuation Axis", self.anchor_axis_combo)
            config_form.addRow("Actuation Line Ratio", self.anchor_line_spin)
            config_form.addRow("Finalize Quiet ms", self.finalize_quiet_spin)
            config_form.addRow("Nozzle Distance mm", self.nozzle_distance_spin)
            config_form.addRow("Belt Speed mm/s", self.belt_speed_spin)
            config_form.addRow("Trigger Offset s", self.trigger_offset_spin)
            config_form.addRow("Latency Compensation ms", self.latency_compensation_spin)
            config_form.addRow("Timing Log Dir", self.timing_log_dir_input)
            config_form.addRow("Debug Dir", self.debug_dir_input)
            config_form.addRow("Pictures Dir", self.pictures_dir_input)
            config_form.addRow("", self.simulate_gpio_checkbox)
            side_layout.addWidget(config_group, 1)

            recent_group = base_ui.QGroupBox("Recent Events")
            recent_layout = base_ui.QVBoxLayout(recent_group)
            self.recent_events_table = base_ui.QTableWidget(0, 6)
            self.recent_events_table.setHorizontalHeaderLabels(
                ["Recorded", "Event", "Result", "Class", "Score", "Source"]
            )
            self.recent_events_table.setEditTriggers(
                base_ui.QTableWidget.EditTrigger.NoEditTriggers
            )
            self.recent_events_table.setSelectionBehavior(
                base_ui.QTableWidget.SelectionBehavior.SelectRows
            )
            self.recent_events_table.verticalHeader().setVisible(False)
            recent_layout.addWidget(self.recent_events_table)
            layout.addWidget(recent_group, 1)

            timing_group = base_ui.QGroupBox("Timing Review")
            timing_layout = base_ui.QVBoxLayout(timing_group)
            self.timing_log_path_label = base_ui.QLabel(DEFAULT_TIMING_LOG_DIR)
            timing_layout.addWidget(self.timing_log_path_label)
            self.timing_table = base_ui.QTableWidget(0, 8)
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
            self.timing_table.setEditTriggers(
                base_ui.QTableWidget.EditTrigger.NoEditTriggers
            )
            self.timing_table.setSelectionBehavior(
                base_ui.QTableWidget.SelectionBehavior.SelectRows
            )
            self.timing_table.verticalHeader().setVisible(False)
            timing_layout.addWidget(self.timing_table)
            layout.addWidget(timing_group, 1)

        def _build_runtime_args(self):
            args = cap_line_runtime_v2.parse_args([])
            args.no_display = True
            args.model = self.model_input.text().strip() or DEFAULT_MODEL
            args.cams = [
                self.cam0_input.text().strip() or "0",
                self.cam1_input.text().strip() or "1",
            ]
            args.res = [self.width_spin.value(), self.height_spin.value()]
            args.fps = self.fps_spin.value()
            args.pixel_format = cap_line_runtime_v2.DEFAULT_CAMERA_PIXEL_FORMAT
            args.exposure = self.exposure_spin.value()
            args.tracking_threshold = self.tracking_threshold_spin.value()
            args.reject_threshold = self.reject_threshold_spin.value()
            args.global_threshold = args.tracking_threshold
            args.pair_max_skew_ms = self.pair_skew_spin.value()
            args.trigger_pin = (
                self.trigger_pin_input.text().strip()
                or cap_line_runtime_v2.DEFAULT_TRIGGER_PIN
            )
            args.trigger_duration = self.trigger_duration_spin.value()
            args.trigger_min_gap = self.trigger_gap_spin.value()
            args.timing_camera = int(self.timing_camera_combo.currentText())
            args.anchor_axis = self.anchor_axis_combo.currentText()
            args.anchor_line_ratio = self.anchor_line_spin.value()
            args.finalize_quiet_ms = self.finalize_quiet_spin.value()
            args.nozzle_distance_mm = self.nozzle_distance_spin.value()
            args.belt_speed_mm_per_s = self.belt_speed_spin.value()
            args.trigger_offset_s = self.trigger_offset_spin.value()
            args.latency_compensation_ms = self.latency_compensation_spin.value()
            args.timing_log_dir = (
                self.timing_log_dir_input.text().strip() or DEFAULT_TIMING_LOG_DIR
            )
            args.debug_dir = (
                self.debug_dir_input.text().strip() or DEFAULT_DEBUG_DIR
            )
            args.pictures_dir = (
                self.pictures_dir_input.text().strip() or DEFAULT_PICTURES_DIR
            )
            args.review_dir = args.debug_dir
            args.session_log_dir = cap_line_runtime_v2.DEFAULT_SESSION_LOG_DIR
            args.simulate_gpio = self.simulate_gpio_checkbox.isChecked()
            return args

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

            if changes.get("latest_performance") is not None:
                self._update_performance_metrics(changes["latest_performance"])

            if changes["error"] is not None or changes["stopped"]:
                self._sync_controls()

            if self._closing_after_stop and not self.controller.is_running:
                self._closing_after_stop = False
                self.close()

        def _update_performance_metrics(self, snapshot) -> None:
            latest_skew = getattr(snapshot, "latest_pair_skew_ms", None)
            dropped_pairs = getattr(snapshot, "stale_pair_drops", 0)
            self.metric_labels["pair_skew"].setText(
                "-" if latest_skew is None else f"{float(latest_skew):.1f} ms"
            )
            self.metric_labels["dropped_pairs"].setText(str(int(dropped_pairs)))


    def main() -> None:
        app = base_ui.QApplication([])
        window = DetectionApp()
        window.show()
        app.exec()


else:
    class DetectionApp:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyQt6 is required to use cap_line_ui_v2_grey.py")


    def main() -> None:
        raise RuntimeError("PyQt6 is required to run cap_line_ui_v2_grey.py")


if __name__ == "__main__":
    main()
