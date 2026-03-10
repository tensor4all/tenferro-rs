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

        self.assertEqual(len(records), 24)
        self.assertEqual(records[-1]["case_id"], "solve_f64_identity_024")

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

        self.assertEqual(len(records), 1)
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
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["case_id"], record["case_id"])


if __name__ == "__main__":
    unittest.main()
