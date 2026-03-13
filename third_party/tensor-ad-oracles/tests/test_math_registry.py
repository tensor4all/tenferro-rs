import json
import tempfile
import unittest
from pathlib import Path

from validators import math_registry


class MathRegistryTests(unittest.TestCase):
    def _make_repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        tmpdir = tempfile.TemporaryDirectory()
        root = Path(tmpdir.name)
        (root / "docs" / "math").mkdir(parents=True)
        (root / "cases").mkdir(parents=True)
        return tmpdir, root

    def _write_registry(self, root: Path, entries: list[dict]) -> None:
        (root / "docs" / "math" / "registry.json").write_text(
            json.dumps({"version": 1, "entries": entries}, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_extract_markdown_anchors_supports_explicit_ids(self) -> None:
        anchors = math_registry.extract_markdown_anchors(
            "\n".join(
                [
                    "<a id=\"family-identity\"></a>",
                    "### `identity`",
                    "",
                    "### `u_abs` {#family-u-abs}",
                ]
            )
        )

        self.assertEqual(anchors, {"family-identity", "family-u-abs"})

    def test_validate_registry_accepts_minimal_valid_repo(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "solve.md").write_text(
            "# Solve\n\n## DB Families\n\n<a id=\"family-identity\"></a>\n",
            encoding="utf-8",
        )
        solve_dir = root / "cases" / "solve"
        solve_dir.mkdir()
        (solve_dir / "identity.jsonl").write_text("{}\n", encoding="utf-8")
        self._write_registry(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note_path": "docs/math/solve.md",
                    "anchor": "family-identity",
                }
            ],
        )

        math_registry.validate_registry(root)

    def test_validate_registry_rejects_duplicate_entries(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "solve.md").write_text(
            "<a id=\"family-identity\"></a>\n",
            encoding="utf-8",
        )
        self._write_registry(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note_path": "docs/math/solve.md",
                    "anchor": "family-identity",
                },
                {
                    "op": "solve",
                    "family": "identity",
                    "note_path": "docs/math/solve.md",
                    "anchor": "family-identity",
                },
            ],
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            math_registry.validate_registry(root)

    def test_validate_registry_rejects_missing_note_path(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        self._write_registry(
            root,
            [
                {
                    "op": "solve",
                    "family": "identity",
                    "note_path": "docs/math/missing.md",
                    "anchor": "family-identity",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "note_path"):
            math_registry.validate_registry(root)

    def test_validate_registry_rejects_missing_anchor(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "svd.md").write_text(
            "# SVD\n\n## DB Families\n",
            encoding="utf-8",
        )
        self._write_registry(
            root,
            [
                {
                    "op": "svd",
                    "family": "u_abs",
                    "note_path": "docs/math/svd.md",
                    "anchor": "family-u-abs",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "missing anchor"):
            math_registry.validate_registry(root)

    def test_validate_registry_rejects_missing_case_coverage(self) -> None:
        tmpdir, root = self._make_repo()
        self.addCleanup(tmpdir.cleanup)

        (root / "docs" / "math" / "svd.md").write_text(
            "<a id=\"family-u-abs\"></a>\n",
            encoding="utf-8",
        )
        svd_dir = root / "cases" / "svd"
        svd_dir.mkdir()
        (svd_dir / "u_abs.jsonl").write_text("{}\n", encoding="utf-8")
        self._write_registry(root, [])

        with self.assertRaisesRegex(ValueError, "missing registry entries"):
            math_registry.validate_registry(root)

    def test_repo_contains_core_linalg_math_notes(self) -> None:
        note_dir = Path(__file__).resolve().parents[1] / "docs" / "math"
        expected = {
            "svd.md",
            "solve.md",
            "qr.md",
            "lu.md",
            "cholesky.md",
            "inv.md",
            "det.md",
            "eig.md",
            "eigen.md",
            "pinv.md",
            "lstsq.md",
            "norm.md",
        }

        self.assertTrue(expected.issubset({path.name for path in note_dir.glob("*.md")}))

    def test_repo_svd_note_exposes_family_anchors(self) -> None:
        note_path = Path(__file__).resolve().parents[1] / "docs" / "math" / "svd.md"
        text = note_path.read_text(encoding="utf-8")
        anchors = math_registry.extract_markdown_anchors(text)

        self.assertIn("## DB Families", text)
        self.assertEqual(
            {"family-u-abs", "family-s", "family-vh-abs", "family-uvh-product"} - anchors,
            set(),
        )

    def test_repo_svd_note_retains_nonlossy_core_derivation_details(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "svd.md"
        ).read_text(encoding="utf-8")

        self.assertIn("F_{ij}", text)
        self.assertIn("S_{\\text{inv},i}", text)
        self.assertIn("\\Gamma_{\\bar{U}}", text)
        self.assertIn("\\Gamma_{\\bar{V}}", text)
        self.assertIn("Non-square corrections", text)
        self.assertIn("gauge", text)

    def test_repo_qr_note_retains_nonlossy_case_split_and_helpers(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "qr.md"
        ).read_text(encoding="utf-8")

        self.assertIn("copyltu", text)
        self.assertIn("Full-rank", text)
        self.assertIn("Wide Reduced QR", text)
        self.assertIn("R^{-\\dagger}", text)
        self.assertIn("trilImInvAdjSkew", text)
        self.assertIn("Triangular solve", text)

    def test_repo_lu_note_retains_case_split_and_triangular_helpers(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "lu.md"
        ).read_text(encoding="utf-8")

        self.assertIn("\\mathrm{tril}_-(X)", text)
        self.assertIn("Square case", text)
        self.assertIn("Wide case", text)
        self.assertIn("Tall case", text)
        self.assertIn("triangular solves", text)

    def test_repo_eig_note_retains_gap_and_normalization_details(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "eig.md"
        ).read_text(encoding="utf-8")

        self.assertIn("V^{-1}\\dot{A}\\,V", text)
        self.assertIn("Normalization correction", text)
        self.assertIn("Gauge invariance check", text)
        self.assertIn("values_vectors_abs", text)

    def test_repo_eigen_note_retains_hermitian_inner_matrix_details(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "eigen.md"
        ).read_text(encoding="utf-8")

        self.assertIn("Step 2: Build the inner matrix $D$", text)
        self.assertIn("\\bar{A} = U D U^\\dagger", text)
        self.assertIn("symmetrization", text)
        self.assertIn("gauge_ill_defined", text)

    def test_repo_solve_note_retains_triangular_and_right_solve_details(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "solve.md"
        ).read_text(encoding="utf-8")

        self.assertIn("A^{-\\mathsf{H}}", text)
        self.assertIn("Right-side solve", text)
        self.assertIn("unit-triangular", text)
        self.assertIn("lu_solve", text)

    def test_repo_lstsq_note_retains_qr_residual_derivation(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "lstsq.md"
        ).read_text(encoding="utf-8")

        self.assertIn("y = R^{-\\dagger} \\bar{x}", text)
        self.assertIn("z = R^{-1} y", text)
        self.assertIn("r = b - Ax", text)
        self.assertIn("gradient-oriented upstream variant", text)

    def test_repo_pinv_note_retains_projector_and_three_term_formulas(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "pinv.md"
        ).read_text(encoding="utf-8")

        self.assertIn("P_{\\mathrm{col}}", text)
        self.assertIn("P_{\\mathrm{row}}", text)
        self.assertIn("Three-term interpretation", text)
        self.assertIn("Golub", text)

    def test_repo_norm_note_retains_vector_matrix_and_spectral_cases(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "norm.md"
        ).read_text(encoding="utf-8")

        self.assertIn("Vector $p$-norm", text)
        self.assertIn("Frobenius norm", text)
        self.assertIn("Nuclear norm", text)
        self.assertIn("Spectral norm", text)

    def test_repo_cholesky_note_retains_phi_operator_details(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "cholesky.md"
        ).read_text(encoding="utf-8")

        self.assertIn("\\varphi(X) = \\mathrm{tril}(X) - \\tfrac{1}{2}\\mathrm{diag}(X)", text)
        self.assertIn("\\varphi^*", text)
        self.assertIn("L^{-1}\\dot{A}\\,L^{-\\mathsf{H}}", text)

    def test_repo_inv_note_retains_solve_relationship(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "inv.md"
        ).read_text(encoding="utf-8")

        self.assertIn("\\dot{B} = -B\\,\\dot{A}\\,B", text)
        self.assertIn("Relationship to solve", text)
        self.assertIn("B^{\\mathsf{H}}\\,\\bar{B}\\,B^{\\mathsf{H}}", text)

    def test_repo_det_note_retains_slogdet_and_singular_handling(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "det.md"
        ).read_text(encoding="utf-8")

        self.assertIn("Jacobi", text)
        self.assertIn("Singular matrix handling", text)
        self.assertIn("slogdet", text)
        self.assertIn("orientation/phase factor", text)

    def test_repo_matrix_exp_note_retains_block_matrix_and_pytorch_mapping(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "matrix_exp.md"
        ).read_text(encoding="utf-8")

        self.assertIn("Mathias 1996", text)
        self.assertIn("2N \\times 2N", text)
        self.assertIn("differential_analytic_matrix_function", text)
        self.assertIn("Computational cost", text)

    def test_repo_dyadtensor_reverse_note_retains_pullback_bridge_details(self) -> None:
        text = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "math"
            / "dyadtensor_reverse.md"
        ).read_text(encoding="utf-8")

        self.assertIn("register a local pullback", text)
        self.assertIn("ad::pullback_wrt_mixed", text)
        self.assertIn("eig_ad(...).run()", text)
        self.assertIn("register_bridge_rule", text)

    def test_repo_scalar_ops_note_retains_pytorch_baseline_and_reduction_wrappers(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "scalar_ops.md"
        ).read_text(encoding="utf-8")

        self.assertIn("PyTorch Baseline", text)
        self.assertIn("handle_r_to_c", text)
        self.assertIn("mean_ad", text)
        self.assertIn("var_ad", text)
        self.assertIn("std_ad", text)
        self.assertIn("powf", text)
        self.assertIn("powi", text)
        self.assertIn("Tensor-Composite Rules", text)

    def test_repo_eig_and_eigen_notes_are_distinct(self) -> None:
        note_dir = Path(__file__).resolve().parents[1] / "docs" / "math"
        eig_text = (note_dir / "eig.md").read_text(encoding="utf-8")
        eigen_text = (note_dir / "eigen.md").read_text(encoding="utf-8")

        self.assertIn("General", eig_text)
        self.assertIn("Hermitian", eigen_text)

    def test_repo_contains_remaining_known_rule_notes(self) -> None:
        note_dir = Path(__file__).resolve().parents[1] / "docs" / "math"
        expected = {"matrix_exp.md", "scalar_ops.md", "dyadtensor_reverse.md"}

        self.assertTrue(expected.issubset({path.name for path in note_dir.glob("*.md")}))

    def test_repo_scalar_ops_note_exposes_representative_op_anchors(self) -> None:
        note_path = Path(__file__).resolve().parents[1] / "docs" / "math" / "scalar_ops.md"
        anchors = math_registry.extract_markdown_anchors(note_path.read_text(encoding="utf-8"))

        self.assertEqual({"op-abs", "op-add", "op-sum", "op-var"} - anchors, set())

    def test_repo_matrix_exp_note_marks_db_status(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "docs" / "math" / "matrix_exp.md"
        ).read_text(encoding="utf-8")

        self.assertIn("not yet materialized", text)

    def test_repo_registry_contains_representative_family_mappings(self) -> None:
        root = Path(__file__).resolve().parents[1]
        entries = math_registry.load_registry(root)["entries"]
        index = {(row["op"], row["family"]): row for row in entries}

        self.assertEqual(index[("svd", "u_abs")]["note_path"], "docs/math/svd.md")
        self.assertEqual(index[("svd", "u_abs")]["anchor"], "family-u-abs")
        self.assertEqual(index[("eig", "values_vectors_abs")]["note_path"], "docs/math/eig.md")
        self.assertEqual(
            index[("eig", "values_vectors_abs")]["anchor"],
            "family-values-vectors-abs",
        )
        self.assertEqual(index[("solve", "identity")]["note_path"], "docs/math/solve.md")
        self.assertEqual(index[("solve", "identity")]["anchor"], "family-solve-identity")
        self.assertEqual(index[("abs", "identity")]["note_path"], "docs/math/scalar_ops.md")
        self.assertEqual(index[("abs", "identity")]["anchor"], "op-abs")
        self.assertEqual(index[("sum", "identity")]["note_path"], "docs/math/scalar_ops.md")
        self.assertEqual(index[("sum", "identity")]["anchor"], "op-sum")

    def test_repo_registry_covers_materialized_case_families(self) -> None:
        math_registry.validate_registry(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
