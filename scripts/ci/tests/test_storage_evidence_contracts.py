import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FreezeEvidenceTests(unittest.TestCase):
    def test_only_closed_evidence_paths_are_accepted(self) -> None:
        module = load_script("check-storage-contract-freeze.py")
        module.validate_evidence_paths({"docs/testing/storage-hardware-matrix.md"})
        with self.assertRaisesRegex(module.CheckError, "non-evidence path"):
            module.validate_evidence_paths({"crates/tenferro-tensor/src/types.rs"})

    def test_refresh_ignores_saved_reports(self) -> None:
        saved = {"candidate_commit": "a" * 40, "status": "pass", "result": "pass"}
        for script in (
            "check-storage-contract-freeze.py",
            "check-storage-static-rank-codegen.py",
            "verify-storage-traversal-performance.py",
        ):
            with self.subTest(script=script):
                module = load_script(script)
                self.assertIsNone(module.select_existing_record(saved, refresh=True))


class ClosureReproductionTests(unittest.TestCase):
    def test_reproduction_command_set_is_bounded(self) -> None:
        module = load_script("check-storage-redesign-closure.py")
        self.assertEqual(
            [item[0] for item in module.REPRODUCE_COMMANDS],
            [
                "p10-api-normalization",
                "p4-traversal-resolution-counts",
                "p3-static-rank-preservation",
                "p3-host-owner",
                None,
                None,
            ],
        )

    def test_receipt_checker_failure_stops_before_execution(self) -> None:
        module = load_script("check-storage-redesign-closure.py")
        ran = False

        def runner(argv: tuple[str, ...]) -> int:
            nonlocal ran
            ran = True
            return 0

        with self.assertRaisesRegex(module.CheckError, "receipt checker"):
            module.run_reproduction(
                Path("receipt.json"),
                receipt_validator=lambda _: 1,
                runner=runner,
            )
        self.assertFalse(ran)

    def test_nonzero_reproduction_fails(self) -> None:
        module = load_script("check-storage-redesign-closure.py")

        def runner(argv: tuple[str, ...]) -> int:
            return 1 if argv[0] == "python3" else 0

        with self.assertRaisesRegex(module.CheckError, "exit code 1"):
            module.run_reproduction(
                Path("receipt.json"),
                receipt_validator=lambda _: 0,
                runner=runner,
            )


class HardwareMatrixTests(unittest.TestCase):
    candidate = "a" * 40

    def partial(self, names: tuple[str, ...], *, candidate: str | None = None) -> dict:
        return {
            "schema": "tenferro.storage-hardware-matrix.v1",
            "candidate_commit": candidate or self.candidate,
            "complete": False,
            "lanes": [
                {
                    "lane": name,
                    "status": "pass",
                    "command": f"run-{name}",
                    "environment": "test-host",
                    "device_facts": f"test-{name}",
                    "test_count": 1,
                    "passed": 1,
                    "failed": 0,
                    "ignored": 0,
                    "evidence": f"tests/{name}.rs",
                    "skip_reason": None,
                }
                for name in names
            ],
        }

    def test_merge_accepts_one_candidate_and_all_required_lanes(self) -> None:
        module = load_script("check-storage-hardware-matrix.py")
        merged = module.merge_records(
            self.candidate,
            [
                self.partial(("cpu", "cuda2", "cuda-ad")),
                self.partial(("webgpu", "metal")),
            ],
        )
        self.assertTrue(merged["complete"])
        self.assertEqual(merged["status"], "pass")
        self.assertEqual(
            [lane["lane"] for lane in merged["lanes"]], list(module.REQUIRED)
        )

    def test_merge_rejects_mismatch_duplicate_missing_and_skip(self) -> None:
        module = load_script("check-storage-hardware-matrix.py")
        mismatch = [
            self.partial(("cpu", "cuda2", "cuda-ad")),
            self.partial(("webgpu", "metal"), candidate="b" * 40),
        ]
        duplicate = [
            self.partial(("cpu", "cuda2", "cuda-ad")),
            self.partial(("cpu", "webgpu", "metal")),
        ]
        missing = [self.partial(("cpu", "cuda2", "cuda-ad", "webgpu"))]
        skipped = [
            self.partial(("cpu", "cuda2", "cuda-ad")),
            self.partial(("webgpu", "metal")),
        ]
        skipped[1]["lanes"][1]["status"] = "skip"
        skipped[1]["lanes"][1]["test_count"] = 0
        for name, records in (
            ("candidate", mismatch),
            ("duplicate", duplicate),
            ("missing", missing),
            ("skip", skipped),
        ):
            with self.subTest(name=name):
                with self.assertRaises(module.CheckError):
                    module.merge_records(self.candidate, records)


if __name__ == "__main__":
    unittest.main()
