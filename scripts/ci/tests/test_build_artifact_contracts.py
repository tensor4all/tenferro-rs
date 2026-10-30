from __future__ import annotations

import os
import re
import runpy
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]


def dependency_spec(value: object) -> dict[str, object]:
    """Normalize a dependency that may be spelled as a bare version string."""
    return value if isinstance(value, dict) else {"version": value}
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
    def test_repository_cargo_config_disables_dev_debug_info_by_default(self) -> None:
        config = tomllib.loads((ROOT / ".cargo" / "config.toml").read_text())
        self.assertEqual(
            config["env"]["CARGO_PROFILE_DEV_DEBUG"],
            {"value": "0", "force": False},
        )

    def test_guide_cargo_environment_defaults_dev_debug_info_to_zero(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            script = runpy.run_path(
                str(ROOT / "scripts" / "check-guide-dependency-snippets.py")
            )
            env = script["cargo_environment"](Path("/tmp/target"))

        self.assertEqual(env["CARGO_PROFILE_DEV_DEBUG"], "0")

    def test_guide_cargo_environment_preserves_dev_debug_info_override(self) -> None:
        with patch.dict(
            os.environ, {"CARGO_PROFILE_DEV_DEBUG": "2"}, clear=True
        ):
            script = runpy.run_path(
                str(ROOT / "scripts" / "check-guide-dependency-snippets.py")
            )
            env = script["cargo_environment"](Path("/tmp/target"))

        self.assertEqual(env["CARGO_PROFILE_DEV_DEBUG"], "2")

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
                explicit_test_sources = sorted(
                    Path(target["path"]).name
                    for target in manifest.get("test", [])
                    if target.get("path", "").count("/") == 1
                )
                self.assertEqual(top_level_sources, explicit_test_sources)

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

    def test_workspace_faer_and_strided_dependencies_follow_contract(self) -> None:
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        dependencies = manifest["workspace"]["dependencies"]

        faer = dependencies["faer"]
        self.assertFalse(faer["default-features"])
        self.assertEqual(set(faer["features"]), {"std", "rayon", "threadteam"})
        self.assertRegex(faer["version"], r"^\^?\d+\.\d+\.\d+$")

        # The registry strided release line the CPU contraction/linalg stack is
        # developed and tested against. Requirements are caret ranges at the
        # patch each API needs, never exact pins and never git: Cargo unifies
        # one compatible line to one version, and the committed `Cargo.lock`
        # fixes the exact versions a build uses. The requirement is asserted by
        # shape, not by number, so a bump does not edit this test.
        strided = (
            "strided-view",
            "strided-traits",
            "strided-perm",
            "strided-basic",
            "strided-fused",
            "strided-kernel",
        )
        for name in strided:
            with self.subTest(dependency=name):
                spec = dependency_spec(dependencies[name])
                self.assertNotIn("git", spec)
                version = spec["version"]
                self.assertTrue(
                    isinstance(version, str) and version[:1].isdigit(),
                    f"{name} must be a caret registry requirement, got {version!r}",
                )
        # The family publishes its crates on independent lines (`strided-view` and
        # `strided-traits` stay at 0.7 while the kernel crates are at 0.8), so the
        # requirements are not one release line. The invariant is that the provider
        # crates (cpueinsum, tlinalg) meet this crate's `StridedView` at a crate
        # boundary, which needs exactly one copy of each strided crate in the
        # resolved graph: the committed lock must hold one version per name.
        lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
        resolved: dict[str, list[str]] = {}
        for package in lock["package"]:
            if package["name"].startswith("strided-"):
                resolved.setdefault(package["name"], []).append(package["version"])
        for name in strided:
            with self.subTest(locked=name):
                self.assertEqual(
                    len(resolved.get(name, [])),
                    1,
                    f"{name} must resolve to exactly one version, got {resolved.get(name)}",
                )
        for name, versions in resolved.items():
            with self.subTest(locked_duplicate=name):
                self.assertEqual(len(versions), 1, f"duplicate {name}: {versions}")

        self.assertNotIn("strided-einsum2", dependencies)

        # The provider crates that take this crate's views follow the same shape:
        # registry caret requirements, so one strided source resolves in the
        # graph; the committed lock records which versions.
        providers = (
            ("Cargo.toml", ("cpueinsum", "cpueinsum-blas", "tlinalg", "tlinalg-blas")),
        )
        for relative, names in providers:
            manifest = tomllib.loads((ROOT / relative).read_text())
            section = manifest.get("workspace", manifest).get("dependencies", {})
            for name in names:
                with self.subTest(provider=name):
                    spec = dependency_spec(section[name])
                    self.assertNotIn("git", spec)
                    self.assertTrue(
                        isinstance(spec["version"], str)
                        and spec["version"][:1].isdigit(),
                        f"{name} must be a caret registry requirement",
                    )

        # The lockfile is what makes the resolved graph reproducible.
        self.assertTrue((ROOT / "Cargo.lock").is_file())

    def test_linalg_provider_dependencies_are_isolated(self) -> None:
        manifest = tomllib.loads(
            (ROOT / "crates" / "tenferro-linalg" / "Cargo.toml").read_text()
        )
        features = manifest["features"]
        dependencies = manifest["dependencies"]

        self.assertIn("dep:faer", features["native"])
        self.assertIn("dep:lapack", features["blas"])
        self.assertTrue(dependencies["faer"]["optional"])
        self.assertTrue(dependencies["lapack"]["optional"])

    def test_cuda_dependencies_are_pinned_revisions(self) -> None:
        """The CUDA engine's two upstream crates are exact revisions of their releases.

        A branch or a loose version would let the tested kernel ABI change under the
        build, so both stay exact revisions in `[workspace.dependencies]`, next to the
        exact crates.io version `cargo publish` keeps (the release preflight and
        `check-git-pin-content.py` scan that table), and `tenferro-gpu` takes them as
        optional workspace dependencies. `cudarc` keeps one feature set so a single
        CUDA binding resolves in the graph.
        """
        workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"][
            "dependencies"
        ]
        gpu = tomllib.loads(
            (ROOT / "crates" / "tenferro-gpu" / "Cargo.toml").read_text()
        )["dependencies"]
        for name in ("t4a-cuda-runtime", "t4a-cuda-kernels"):
            with self.subTest(dependency=name):
                spec = workspace[name]
                self.assertIn("git", spec)
                self.assertNotIn("branch", spec)
                self.assertNotIn("tag", spec)
                self.assertEqual(len(spec["rev"]), 40, "an exact revision")
                self.assertRegex(spec["version"], r"^=\d+\.\d+\.\d+$")
                self.assertEqual(gpu[name], {"workspace": True, "optional": True})

        cudarc = workspace["cudarc"]
        self.assertFalse(cudarc["default-features"])
        self.assertEqual(
            set(cudarc["features"]),
            {"driver", "runtime", "nvrtc", "dynamic-loading", "cuda-12080"},
        )


if __name__ == "__main__":
    unittest.main()
