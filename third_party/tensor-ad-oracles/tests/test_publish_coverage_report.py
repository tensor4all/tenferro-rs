import tempfile
import unittest
from pathlib import Path

from scripts import report_upstream_publish_coverage


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_REPORT = (
    REPO_ROOT / "docs" / "generated" / "pytorch-upstream-publish-coverage.md"
)


class PublishCoverageReportTests(unittest.TestCase):
    def test_build_report_includes_expected_sections(self) -> None:
        report = report_upstream_publish_coverage.build_report_text()

        self.assertIn("# PyTorch Upstream Publish Coverage", report)
        self.assertIn("## Upstream Inventory", report)
        self.assertIn("## Publishable Family Coverage", report)
        self.assertIn("## Missing Publishable Coverage", report)

    def test_build_report_highlights_representative_svd_publish_coverage(self) -> None:
        report = report_upstream_publish_coverage.build_report_text()

        self.assertIn("| svd | u_abs | success |", report)
        self.assertIn("float64, complex128, float32, complex64", report)
        self.assertIn(
            "| svd | u_abs | success | float64, complex128, float32, complex64 | "
            "float64, complex128, float32, complex64 | - |",
            report,
        )
        self.assertIn("## Missing Publishable Coverage\n\nNone.", report)

    def test_main_writes_report_and_matches_checked_in_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "coverage.md"

            self.assertEqual(
                report_upstream_publish_coverage.main(["--output", str(output_path)]),
                0,
            )
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                CHECKED_IN_REPORT.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
