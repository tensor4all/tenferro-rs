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

    def test_workspace_faer_dependency_disables_broad_defaults(self) -> None:
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        dependencies = manifest["workspace"]["dependencies"]

        faer = dependencies["faer"]
        self.assertFalse(faer["default-features"])
        self.assertEqual(set(faer["features"]), {"std", "rayon"})

        revision = "017c7e2413e48e5182590eed9b2e99350cbd5283"
        for name in (
            "strided-view",
            "strided-traits",
            "strided-perm",
            "strided-kernel",
            "strided-einsum2",
        ):
            with self.subTest(dependency=name):
                self.assertEqual(dependencies[name]["rev"], revision)

        self.assertFalse(dependencies["strided-einsum2"]["default-features"])

if __name__ == "__main__":
    unittest.main()
