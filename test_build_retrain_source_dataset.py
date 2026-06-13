from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "script" / "build_retrain_source_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_retrain_source_dataset",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildRetrainSourceDatasetTests(unittest.TestCase):
    def test_clean_gate_prefers_centered_complete_caps(self) -> None:
        module = load_module()

        centered = module.LabelRow(
            class_id=0,
            x_center=0.50,
            y_center=0.52,
            width=0.30,
            height=0.60,
        )
        off_center = module.LabelRow(
            class_id=0,
            x_center=0.78,
            y_center=0.52,
            width=0.30,
            height=0.60,
        )
        touches_left_edge = module.LabelRow(
            class_id=0,
            x_center=0.10,
            y_center=0.52,
            width=0.20,
            height=0.60,
        )

        self.assertTrue(module.sample_passes_clean_gate(centered))
        self.assertFalse(module.sample_passes_clean_gate(off_center))
        self.assertFalse(module.sample_passes_clean_gate(touches_left_edge))

    def test_plan_balanced_samples_duplicates_dirt_round_robin(self) -> None:
        module = load_module()

        clean_samples = [
            self._sample(module, f"clean_{index}", 0, 0.50, 0.52, 0.30, 0.60)
            for index in range(5)
        ]
        dirt_samples = [
            self._sample(module, "dirt_a", 1, 0.49, 0.50, 0.32, 0.61),
            self._sample(module, "dirt_b", 1, 0.51, 0.51, 0.31, 0.62),
        ]

        planned = module.plan_balanced_samples(clean_samples, dirt_samples)

        clean_planned = [sample for sample in planned if sample.row.class_id == 0]
        dirt_planned = [sample for sample in planned if sample.row.class_id == 1]
        duplicate_planned = [sample for sample in dirt_planned if sample.is_duplicate]

        self.assertEqual(5, len(clean_planned))
        self.assertEqual(5, len(dirt_planned))
        self.assertEqual(["dirt_a", "dirt_b", "dirt_a"], [sample.source_stem for sample in duplicate_planned])
        self.assertEqual([1, 1, 2], [sample.duplicate_index for sample in duplicate_planned])
        self.assertEqual(len({sample.output_stem for sample in planned}), len(planned))

    def test_main_builds_balanced_dataset_and_metadata(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw_dataset"
            output = root / "raw_dataset_balanced_retrain_v1"
            source.mkdir()

            self._write_sample(source, "clean_keep_1", "0 0.50 0.52 0.30 0.60\n")
            self._write_sample(source, "clean_keep_2", "0 0.45 0.50 0.28 0.55\n")
            self._write_sample(source, "clean_keep_3", "0 0.70 0.60 0.20 0.50\n")
            self._write_sample(source, "clean_reject_geometry", "0 0.86 0.52 0.20 0.50\n")
            self._write_sample(source, "dirt_1", "1 0.49 0.50 0.31 0.61\n")
            self._write_sample(source, "dirt_2", "1 0.50 0.49 0.30 0.60\n")
            self._write_sample(source, "shape_drop", "2 0.50 0.50 0.30 0.60\n")
            self._write_sample(source, "multi_drop", "0 0.50 0.52 0.30 0.60\n1 0.50 0.52 0.30 0.60\n")
            self._write_sample(source, "empty_drop", "")
            (source / "image_without_label.jpg").write_bytes(b"image-without-label")
            (source / "label_without_image.txt").write_text("0 0.50 0.52 0.30 0.60\n", encoding="utf-8")

            exit_code = module.main([str(source), "--output-dir", str(output)])

            self.assertEqual(0, exit_code)
            self.assertTrue(output.exists())

            manifest_path = output / "dataset_manifest.csv"
            summary_path = output / "dataset_summary.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_path.exists())

            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(6, len(rows))
            self.assertEqual(
                {"undefected": 3, "dirt_defect": 3},
                summary["output_class_counts"],
            )
            self.assertEqual(
                {
                    "missing_label": 1,
                    "missing_image": 1,
                    "empty_label": 1,
                    "multiple_objects": 1,
                    "unsupported_class": 1,
                    "clean_filtered_out": 1,
                },
                summary["skipped_reason_counts"],
            )

            duplicate_rows = [row for row in rows if row["is_duplicate"] == "true"]
            self.assertEqual(1, len(duplicate_rows))
            self.assertEqual("dirt_1", duplicate_rows[0]["source_stem"])
            self.assertEqual("1", duplicate_rows[0]["duplicate_index"])

            kept_class_ids = {row["class_id"] for row in rows}
            self.assertEqual({"0", "1"}, kept_class_ids)

            copied_images = sorted(path.name for path in output.glob("*.jpg"))
            copied_labels = sorted(path.name for path in output.glob("*.txt"))
            self.assertEqual(6, len(copied_images))
            self.assertEqual(6, len(copied_labels))

    def _sample(
        self,
        module,
        stem: str,
        class_id: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
    ):
        row = module.LabelRow(
            class_id=class_id,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
        )
        return module.SourceSample(
            stem=stem,
            image_path=Path(f"{stem}.jpg"),
            label_path=Path(f"{stem}.txt"),
            row=row,
        )

    def _write_sample(self, directory: Path, stem: str, label_text: str) -> None:
        (directory / f"{stem}.jpg").write_bytes(b"fake-image-data")
        (directory / f"{stem}.txt").write_text(label_text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
