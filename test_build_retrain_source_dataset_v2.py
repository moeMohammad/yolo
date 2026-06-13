from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "script" / "build_retrain_source_dataset_v2.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_retrain_source_dataset_v2",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildRetrainSourceDatasetV2Tests(unittest.TestCase):
    def test_horizontal_visibility_accepts_full_width_and_rejects_truncated_caps(self) -> None:
        module = load_module()

        fully_visible = module.LabelRow(
            class_id=0,
            x_center=0.50,
            y_center=0.50,
            width=0.30,
            height=0.70,
        )
        clipped_left = module.LabelRow(
            class_id=0,
            x_center=0.09,
            y_center=0.50,
            width=0.20,
            height=0.70,
        )
        clipped_right = module.LabelRow(
            class_id=0,
            x_center=0.93,
            y_center=0.50,
            width=0.16,
            height=0.70,
        )

        self.assertTrue(module.sample_has_full_horizontal_visibility(fully_visible))
        self.assertFalse(module.sample_has_full_horizontal_visibility(clipped_left))
        self.assertFalse(module.sample_has_full_horizontal_visibility(clipped_right))

    def test_stratum_assignment_uses_exact_boundaries(self) -> None:
        module = load_module()

        left_high_small = module.LabelRow(0, 0.32, 0.42, 0.29, 0.60)
        center_mid_medium = module.LabelRow(0, 0.33, 0.43, 0.30, 0.60)
        right_low_large = module.LabelRow(0, 0.68, 0.58, 0.41, 0.60)

        self.assertEqual("left-high-small", module.stratum_id_for_row(left_high_small))
        self.assertEqual("center-mid-medium", module.stratum_id_for_row(center_mid_medium))
        self.assertEqual("right-low-large", module.stratum_id_for_row(right_low_large))

    def test_select_clean_samples_round_robins_across_strata(self) -> None:
        module = load_module()

        clean_samples = [
            self._sample(module, "10", 0, 0.20, 0.40, 0.29, 0.60),  # left-high-small
            self._sample(module, "20", 0, 0.20, 0.40, 0.29, 0.60),  # left-high-small
            self._sample(module, "30", 0, 0.50, 0.50, 0.35, 0.60),  # center-mid-medium
            self._sample(module, "40", 0, 0.50, 0.50, 0.35, 0.60),  # center-mid-medium
            self._sample(module, "50", 0, 0.80, 0.60, 0.45, 0.60),  # right-low-large
        ]

        selected = module.select_clean_samples(clean_samples, target_count=4)

        self.assertEqual(["30", "10", "50", "40"], [sample.stem for sample in selected])
        self.assertEqual(
            [
                "center-mid-medium",
                "left-high-small",
                "right-low-large",
                "center-mid-medium",
            ],
            [module.stratum_id_for_row(sample.row) for sample in selected],
        )

    def test_plan_balanced_samples_duplicates_dirt_in_stratum_aware_order(self) -> None:
        module = load_module()

        clean_selected = [
            self._sample(module, "100", 0, 0.50, 0.50, 0.35, 0.60),
            self._sample(module, "200", 0, 0.20, 0.40, 0.29, 0.60),
            self._sample(module, "300", 0, 0.80, 0.60, 0.45, 0.60),
            self._sample(module, "400", 0, 0.50, 0.60, 0.45, 0.60),
        ]
        dirt_samples = [
            self._sample(module, "11", 1, 0.50, 0.50, 0.35, 0.60),  # center-mid-medium
            self._sample(module, "12", 1, 0.20, 0.40, 0.29, 0.60),  # left-high-small
        ]

        planned = module.plan_balanced_samples(clean_selected, dirt_samples, target_count=4)

        duplicate_rows = [sample for sample in planned if sample.row.class_id == 1 and sample.is_duplicate]
        self.assertEqual(["11", "12"], [sample.source_stem for sample in duplicate_rows])
        self.assertEqual(["1", "1"], [str(sample.duplicate_index) for sample in duplicate_rows])
        self.assertEqual(
            ["center-mid-medium", "left-high-small"],
            [sample.stratum_id for sample in duplicate_rows],
        )

    def test_main_builds_v2_dataset_and_metadata(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw_dataset"
            output = root / "raw_dataset_balanced_retrain_v2"
            source.mkdir()

            self._write_sample(source, "100", "0 0.50 0.50 0.35 0.60\n")  # center-mid-medium
            self._write_sample(source, "200", "0 0.20 0.40 0.29 0.60\n")  # left-high-small
            self._write_sample(source, "300", "0 0.75 0.60 0.45 0.60\n")  # right-low-large
            self._write_sample(source, "400", "0 0.50 0.60 0.45 0.60\n")  # center-low-large
            self._write_sample(source, "450", "0 0.25 0.50 0.35 0.60\n")  # left-mid-medium
            self._write_sample(source, "500", "0 0.09 0.50 0.20 0.60\n")  # clipped clean
            self._write_sample(source, "11", "1 0.50 0.50 0.35 0.60\n")   # center-mid-medium
            self._write_sample(source, "12", "1 0.20 0.40 0.29 0.60\n")   # left-high-small
            self._write_sample(source, "shape_drop", "2 0.50 0.50 0.30 0.60\n")
            self._write_sample(source, "multi_drop", "0 0.50 0.50 0.35 0.60\n1 0.50 0.50 0.35 0.60\n")
            self._write_sample(source, "empty_drop", "")
            (source / "image_without_label.jpg").write_bytes(b"image-without-label")
            (source / "label_without_image.txt").write_text("0 0.50 0.50 0.35 0.60\n", encoding="utf-8")

            exit_code = module.main([str(source), "--output-dir", str(output)])

            self.assertEqual(0, exit_code)
            manifest_path = output / "dataset_manifest.csv"
            summary_path = output / "dataset_summary.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_path.exists())

            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(8, len(rows))
            self.assertEqual(4, summary["target_per_class"])
            self.assertEqual({"undefected": 4, "dirt_defect": 4}, summary["output_class_counts"])
            self.assertEqual(4, summary["clean_selected_count"])
            self.assertEqual(2, summary["dirt_valid_count"])
            self.assertEqual(1, summary["skipped_reason_counts"]["clean_horizontally_clipped"])
            self.assertEqual(1, summary["skipped_reason_counts"]["missing_label"])
            self.assertEqual(1, summary["skipped_reason_counts"]["missing_image"])
            self.assertEqual(1, summary["skipped_reason_counts"]["unsupported_class"])
            self.assertEqual(1, summary["skipped_reason_counts"]["multiple_objects"])
            self.assertEqual(1, summary["skipped_reason_counts"]["empty_label"])

            duplicate_rows = [row for row in rows if row["is_duplicate"] == "true"]
            self.assertEqual(2, len(duplicate_rows))
            self.assertEqual({"center-mid-medium", "left-high-small"}, {row["stratum_id"] for row in duplicate_rows})

            clean_x_bins = {row["x_bin"] for row in rows if row["class_id"] == "0"}
            self.assertEqual({"left", "center"}, clean_x_bins)

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
