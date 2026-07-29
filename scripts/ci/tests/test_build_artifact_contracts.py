from __future__ import annotations

import re
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

                harness = (crate_root / "tests" / "integration.rs").read_text()
                included_paths = set(re.findall(r'#\[path = "([^"]+)"\]', harness))
                suite_root = crate_root / "tests" / "integration"
                expected_paths = {
                    path.relative_to(crate_root / "tests").as_posix()
                    for path in suite_root.glob("*.rs")
                }
                expected_paths.update(
                    path.relative_to(crate_root / "tests").as_posix()
                    for path in suite_root.glob("*/mod.rs")
                )
                self.assertEqual(included_paths, expected_paths)

    def test_ci_does_not_select_removed_integration_targets(self) -> None:
        operational_files = (
            ROOT / "scripts" / "ci" / "run_profile.py",
            ROOT / ".github" / "workflows" / "CI_gpu.yml",
            ROOT / ".github" / "workflows" / "runpod-gpu-test.yml",
        )
        removed_selectors = ("--test inject_tests", "--test pjrt_execution")
        for path in operational_files:
            contents = path.read_text()
            with self.subTest(path=path.relative_to(ROOT)):
                for selector in removed_selectors:
                    self.assertNotIn(selector, contents)

    def test_workspace_faer_dependency_disables_broad_defaults(self) -> None:
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        dependencies = manifest["workspace"]["dependencies"]

        faer = dependencies["faer"]
        self.assertFalse(faer["default-features"])
        self.assertEqual(set(faer["features"]), {"std", "rayon"})

        revision = "649772c8402e5fe95335366326b6623f9a4f5b0a"
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

    def test_linalg_provider_dependencies_are_isolated(self) -> None:
        manifest = tomllib.loads(
            (ROOT / "crates" / "tenferro-linalg" / "Cargo.toml").read_text()
        )
        features = manifest["features"]
        dependencies = manifest["dependencies"]

        self.assertIn("dep:faer", features["cpu-faer"])
        self.assertIn("dep:lapack", features["cpu-blas"])
        self.assertTrue(dependencies["faer"]["optional"])
        self.assertTrue(dependencies["lapack"]["optional"])

    def test_cubecl_dependencies_share_cudarc_contract(self) -> None:
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        dependencies = manifest["workspace"]["dependencies"]

        revision = "346135ab43cececf6405d52a3dbc987537402d27"
        for name in (
            "cubecl",
            "cubecl-cuda",
            "cubecl-common",
            "cubecl-runtime",
            "cubecl-wgpu",
        ):
            with self.subTest(dependency=name):
                self.assertEqual(dependencies[name]["rev"], revision)

        cudarc = dependencies["cudarc"]
        self.assertFalse(cudarc["default-features"])
        self.assertEqual(
            set(cudarc["features"]),
            {"driver", "runtime", "nvrtc", "dynamic-loading", "cuda-12080"},
        )

    def test_cubek_does_not_impose_a_cuda_runtime_floor(self) -> None:
        manifest = tomllib.loads(
            (ROOT / "crates" / "tenferro-gpu" / "Cargo.toml").read_text()
        )
        features = manifest["features"]

        self.assertFalse(any("cubek" in item for item in features["cuda"]))
        self.assertIn("dep:cubek-matmul", features["webgpu"])
        self.assertIn("dep:cubek-std", features["webgpu"])

if __name__ == "__main__":
    unittest.main()
