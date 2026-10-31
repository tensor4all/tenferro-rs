import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ci.change_policy import ChangeClass, classify_paths
from scripts.ci.run_profile import commands_for

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/check-contraction-kernels.py"
spec = importlib.util.spec_from_file_location("contraction_kernels", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

# Exact offending nest from tensor4all/tensor4all-rs PR #1064, tensortrain.rs:470-480
# at 7d180ab11386a899b5d8d2af10f2ccee5c164d5f (MIT); kept as source, not executed Rust.
OFFENDER = """for r in 0..right {
    for s in 0..dim {
        let start = stride * s + next_stride * r;
        let output = &mut result[start..start + stride];
        for l in 0..left {
            let factor = *tensor.get3(l, s, r);
            let input = &scratch[stride * l..stride * (l + 1)];
            for (value, &previous) in output.iter_mut().zip(input) {
                *value = *value + previous * factor;
            }
        }
    }
}"""


def nest(statement="*value += b * c;", iterator="values.iter_mut()", depth=3):
    return "for i in 0..n {\n" * (depth - 1) + f"for value in {iterator} {{\n{statement}\n}}" + "\n}" * (depth - 1)


class ContractionKernelTests(unittest.TestCase):
    def test_real_offending_nest(self):
        self.assertEqual(guard.findings(OFFENDER), [(9, 4)])

    def test_assignment_forms_and_iterators(self):
        for statement in (
            "*value = *value + b * c;",
            "*value += b * c;",
            "*value += b * c",  # a valid unit-valued tail expression
            "*value = *value + b * c",
            "*value = (*value) + (b * c);",
            "*value\n+=\nb\n*\nc;",
            "output[i] += b * c;",
            "sum += *b * *c;",
            "*value += b * -c;",
        ):
            for iterator in ("values.iter_mut()", "a.iter().zip(b)", "&mut values[..]", "&values"):
                with self.subTest(statement=statement, iterator=iterator):
                    self.assertEqual(len(guard.findings(nest(statement, iterator))), 1)

    def test_only_deep_active_loop_nests(self):
        self.assertEqual(guard.findings(nest(depth=2)), [])
        self.assertEqual(len(guard.findings(nest(depth=3))), 1)
        self.assertEqual(guard.findings(nest(depth=1) * 3), [])
        self.assertEqual(guard.findings("if a { match b { _ => {" + nest(depth=1) + "}}}"), [])
        self.assertEqual(len(guard.findings(nest().replace("for i in 0..n", "while ready()"))), 1)
        self.assertEqual(len(guard.findings(nest().replace("for i in 0..n", "loop"))), 1)

    def test_annotation_only_covers_its_enclosing_loop(self):
        for header in ("for r", "    for s", "        for l", "            for (value"):
            annotated = OFFENDER.replace(header, "// contraction-kernel-ok: independent test oracle\n" + header)
            self.assertEqual(guard.findings(annotated), [])
        annotated = "// contraction-kernel-ok: independent test oracle\n" + nest()
        self.assertEqual(len(guard.findings(annotated + "\n" + nest())), 1)

    def test_empty_or_misplaced_annotations_do_not_suppress(self):
        for prefix in (
            "// contraction-kernel-ok:\n",
            "// contraction-kernel-ok:    \n",
            '// contraction-kernel-ok: reason\n\n',
            '// contraction-kernel-ok: reason\nfn f() {\n',
        ):
            self.assertEqual(len(guard.findings(prefix + nest())), 1)
        misplaced = nest().replace("*value +=", "// contraction-kernel-ok: too late\n*value +=")
        self.assertEqual(len(guard.findings(misplaced)), 1)

    def test_legitimate_operations(self):
        for statement in (
            "*value = b * c;",  # overwrite, not accumulation
            "*value = *value * factor;",
            "*value += *b;",  # dereference, not multiplication
            "*value += input[i * stride];",  # address calculation, not product
            "*value += input[offsets[i] * stride];",
            "let value = value + b * c;",  # new binding, not a write
        ):
            self.assertEqual(guard.findings(nest(statement)), [])
        self.assertEqual(guard.findings(nest(iterator="0..n")), [])

    def test_comments_and_literals_are_not_code(self):
        for wrapped in (
            "// " + OFFENDER.replace("\n", "\n// "),
            "/* " + OFFENDER + " /* nested */ */",
            'let s = "' + OFFENDER + '";',
            'let s = r##"' + OFFENDER + '"##;',
            'let s = br#"' + OFFENDER + '"#;',
            "let c = '{'; let d = '}';",
        ):
            self.assertEqual(guard.findings(wrapped), [])
            self.assertEqual(len(guard.findings(wrapped + "\n" + nest())), 1)
        misleading = nest().replace("*value +=", 'let braces = r#" } } } "#; /* } /* } */ } */ *value +=')
        self.assertEqual(len(guard.findings(misleading)), 1)

    def test_exact_file_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "crates/example/src/lib.rs"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text(OFFENDER)
            with patch.object(guard, "ALLOWLIST", {relative: "independent reference oracle"}):
                self.assertEqual(guard.check(root), 0)
            self.assertEqual(guard.ALLOWLIST, {})

    def test_cli_failure_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "crates/example/src/nested/kernel.rs"
            path.parent.mkdir(parents=True)
            path.write_text(OFFENDER)
            command = [sys.executable, str(SCRIPT), "--root", str(root)]
            failed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(failed.returncode, 1, failed.stderr)
            self.assertIn("crates/example/src/nested/kernel.rs:9:", failed.stdout)
            self.assertIn("File allowlist:\n  (empty)", failed.stdout)
            path.unlink()
            for relative in ("crates/example/tests/kernel.rs", "ext/example/src/lib.rs"):
                outside = root / relative
                outside.parent.mkdir(parents=True)
                outside.write_text(OFFENDER)
            path = root / "crates/example/src/lib.rs"
            path.write_text("// no contraction kernel\n")
            passed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            self.assertIn("1 Rust files checked", passed.stdout)

    def test_ci_runs_guard_for_rust_changes(self):
        self.assertIn("python3 scripts/check-contraction-kernels.py", commands_for("ci-config"))
        for relative in ("crates/example/src/lib.rs", "scripts/check-contraction-kernels.py"):
            policy = classify_paths([relative])
            self.assertTrue(policy.run_ci_config)
        policy = classify_paths(["scripts/check-contraction-kernels.py"])
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertFalse(policy.run_gpu)


if __name__ == "__main__":
    unittest.main()
