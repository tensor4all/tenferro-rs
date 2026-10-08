#!/usr/bin/env python3
"""Regression checks for #1777's current-source and dependency boundaries."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from boundary_sources import code_lines, dependency_names


SCRIPTS = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BoundaryTests(unittest.TestCase):
    def test_rust_comments_and_literals_are_not_references(self):
        source = '''// EagerRuntime cubecl
/* outer /* nested */ EagerTensor */
let provider = "cubecl";
let docs = r#"EagerRuntime
cubecl"#;
use cubecl::Runtime;
use tenferro_ad::EagerTensor;
'''
        lines = code_lines(source)
        self.assertNotIn("cubecl", "\n".join(lines[:5]))
        self.assertNotIn("EagerRuntime", "\n".join(lines))
        self.assertIn("cubecl::Runtime", lines[5])
        self.assertIn("EagerTensor", lines[6])

    def test_target_and_renamed_dependencies_are_detected(self):
        manifest = {
            "package": {"description": "cubecl"},
            "dependencies": {"alias": {"package": "cubecl", "version": "1"}},
            "target": {"cfg(unix)": {"build-dependencies": {"cudarc": "1"}}},
            "dev-dependencies": {"tenferro-gpu": "1"},
        }
        self.assertEqual(
            set(dependency_names(manifest)), {"alias", "cubecl", "cudarc", "tenferro-gpu"}
        )

    def test_crate_boundary_checks_code_and_dependencies_not_provider_names(self):
        checker = load("check-crate-boundaries")
        with tempfile.TemporaryDirectory() as directory:
            checker.ROOT = root = Path(directory)
            tensor = root / "crates/tenferro-tensor"
            (tensor / "src").mkdir(parents=True)
            manifest = tensor / "Cargo.toml"
            manifest.write_text('[package]\nname="example"\ndescription="cubecl"\n')
            source = tensor / "src/lib.rs"
            source.write_text('// cubecl\nconst PROVIDER: &str = "cubecl";\n')
            self.assertEqual(checker.check_tensor_has_no_gpu_runtime_deps(), [])
            source.write_text('use cubecl::Runtime;\n')
            self.assertEqual(len(checker.check_tensor_has_no_gpu_runtime_deps()), 1)
            source.write_text('const PROVIDER: &str = "cubecl";\n')
            manifest.write_text('[target.\'cfg(unix)\'.dependencies]\ngpu = {package="cudarc", version="1"}\n')
            self.assertEqual(len(checker.check_tensor_has_no_gpu_runtime_deps()), 1)

    def test_error_docs_only_scan_tracked_live_crates_and_extensions(self):
        checker = load("check-public-error-docs")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            names = ["crates/a/src/lib.rs", "ext/sparse/src/lib.rs", ".worktrees/old/crates/a/src/lib.rs", "docs/plans/old.rs", "crates/a/target/generated.rs"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("pub fn undocumented() -> Result<(), ()> { Ok(()) }\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            self.assertEqual(
                {str(path.relative_to(root)) for path in checker.rust_files(root)},
                {"crates/a/src/lib.rs", "ext/sparse/src/lib.rs"},
            )

    def test_facade_skips_nested_worktrees_and_history_not_live_docs(self):
        checker = load("check-no-facade-crate")
        for relative in [".worktrees/old/README.md", "ext/a/.worktrees/old/README.md", "docs/plans/old.md", "docs/worklogs/old.md"]:
            self.assertTrue(checker.is_skipped(checker.ROOT / relative), relative)
        self.assertFalse(checker.is_skipped(checker.ROOT / "docs/guides/current.md"))

    def test_ad_boundary_retains_feature_and_code_checks(self):
        checker = load("check-ad-boundaries")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checker.__file__ = str(root / "scripts/check-ad-boundaries.py")
            runtime = root / "crates/tenferro-runtime"
            (runtime / "src").mkdir(parents=True)
            manifest = runtime / "Cargo.toml"
            manifest.write_text('[package]\nname="runtime"\ndescription="EagerRuntime"\n')
            source = runtime / "src/lib.rs"
            source.write_text('// EagerRuntime\nconst NOTE: &str = "EagerTensor";\n')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 0)
                source.write_text('use tenferro_ad::EagerRuntime;\n')
                self.assertEqual(checker.main(), 1)
                source.write_text('// EagerRuntime\n')
                manifest.write_text('[features]\nautodiff=[]\n')
                self.assertEqual(checker.main(), 1)

    def test_api_inventory_includes_standalone_extensions(self):
        checker = load("check-api-consistency")
        paths = {crate.path.relative_to(checker.ROOT).as_posix() for crate in checker.workspace_crates(checker.ROOT)}
        self.assertTrue({"ext/sparse", "ext/tropical"} <= paths)


if __name__ == "__main__":
    unittest.main()
