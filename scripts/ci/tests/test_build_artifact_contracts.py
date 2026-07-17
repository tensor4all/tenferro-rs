from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONSOLIDATED_CRATES = (
    "tenferro-ad",
    "tenferro-cpu",
    "tenferro-einsum",
    "tenferro-gpu",
    "tenferro-linalg",
    "tenferro-runtime",
    "tenferro-xla",
)


class BuildArtifactContracts(unittest.TestCase):
    def test_multi_file_integration_suites_link_once_per_crate(self) -> None:
        for crate in CONSOLIDATED_CRATES:
            crate_root = ROOT / "crates" / crate
            with self.subTest(crate=crate):
                manifest = tomllib.loads((crate_root / "Cargo.toml").read_text())
                self.assertFalse(manifest["package"].get("autotests", True))

                integration_targets = [
                    target
                    for target in manifest.get("test", [])
                    if target.get("path") == "tests/integration.rs"
                ]
                self.assertEqual(len(integration_targets), 1)

                top_level_sources = sorted(
                    path.name for path in (crate_root / "tests").glob("*.rs")
                )
                self.assertEqual(top_level_sources, ["integration.rs"])

if __name__ == "__main__":
    unittest.main()
