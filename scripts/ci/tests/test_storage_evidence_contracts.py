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


if __name__ == "__main__":
    unittest.main()
