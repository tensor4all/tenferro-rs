import json
import tempfile
import unittest
from pathlib import Path

from generators import pytorch_v1


class SolveGenerationTests(unittest.TestCase):
    def test_materialize_all_case_families_limit_one_writes_every_family(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = pytorch_v1.materialize_all_case_families(limit=1, cases_root=Path(tmpdir))

            expected = {
                Path(tmpdir) / op / f"{family}.jsonl"
                for op, families in pytorch_v1.build_case_families().items()
                for family in families
            }

            self.assertEqual(set(paths), expected)

    def test_generate_solve_identity_records_returns_all_nonempty_cases(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        records = pytorch_v1.generate_solve_identity_records(limit=24)
        spec = pytorch_v1.build_case_spec_index()[("solve", "identity")]

        self.assertEqual(len(records), 24 * len(spec.supported_dtype_names))
        self.assertEqual(
            {record["dtype"] for record in records},
            set(spec.supported_dtype_names),
        )
        self.assertEqual(records[-1]["case_id"], "solve_c64_identity_024")

    def test_generate_solve_identity_records_is_deterministic(self) -> None:
        try:
            import torch
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        first = pytorch_v1.generate_solve_identity_records(limit=1)
        _ = torch.randn(1024, dtype=torch.float64)
        second = pytorch_v1.generate_solve_identity_records(limit=1)

        self.assertEqual(first, second)

    def test_generate_solve_identity_records_materializes_one_record(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        records = pytorch_v1.generate_solve_identity_records(limit=1)
        spec = pytorch_v1.build_case_spec_index()[("solve", "identity")]

        self.assertEqual(len(records), len(spec.supported_dtype_names))
        record = records[0]
        self.assertEqual(record["op"], "solve")
        self.assertEqual(record["family"], "identity")
        self.assertEqual(len(record["probes"]), 1)
        self.assertIn("first_order", record["comparison"])
        self.assertIn("second_order", record["comparison"])
        self.assertIn("hvp", record["probes"][0]["pytorch_ref"])
        self.assertIn("hvp", record["probes"][0]["fd_ref"])

        spec = pytorch_v1.build_case_spec_index()[("solve", "identity")]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = pytorch_v1.write_case_records(
                spec,
                records,
                cases_root=Path(tmpdir),
            )

            self.assertEqual(out_path, Path(tmpdir) / "solve" / "identity.jsonl")
            saved = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(saved), len(spec.supported_dtype_names))
            self.assertEqual(saved[0]["case_id"], record["case_id"])

    def test_generate_full_pivot_lu_records_cover_rectangular_cases(self) -> None:
        try:
            import torch  # noqa: F401
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        records = pytorch_v1.generate_full_pivot_lu_identity_records(limit=None)
        spec = pytorch_v1.build_case_spec_index()[("full_pivot_lu", "identity")]

        self.assertEqual(len(records), 3 * len(spec.supported_dtype_names))
        self.assertEqual(
            {
                tuple(record["inputs"]["a"]["shape"])
                for record in records
                if record["dtype"] == "float64"
            },
            {(3, 3), (2, 4), (4, 2)},
        )

        record = records[0]
        self.assertEqual(record["op"], "full_pivot_lu")
        self.assertEqual(record["family"], "identity")
        self.assertEqual(record["observable"], {"kind": "identity"})
        self.assertIn("row_perm", record["op_kwargs"])
        self.assertIn("col_perm", record["op_kwargs"])
        self.assertIn("parity", record["op_kwargs"])
        self.assertEqual(record["provenance"]["source_repo"], "tensor-ad-oracles")
        self.assertIn("no upstream full-pivot LU", record["provenance"]["comment"])
        self.assertIn("second_order", record["comparison"])
        self.assertIn("hvp", record["probes"][0]["pytorch_ref"])
        self.assertIn("hvp", record["probes"][0]["fd_ref"])


if __name__ == "__main__":
    unittest.main()
