import os
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.ci.change_policy import ChangeClass, classify_paths
from scripts.ci.run_profile import commands_for

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/check-isa-rules.py"


def run(*args, **env):
    clean = {k: v for k, v in os.environ.items() if "RUSTFLAGS" not in k}
    clean.update(env)
    return subprocess.run([sys.executable, str(SCRIPT), *args], env=clean, text=True, capture_output=True)


class IsaRulesTests(unittest.TestCase):
    def test_self_test_passes(self):
        result = run("--self-test")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_current_tree_passes(self):
        result = run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rustflags_target_cpu_fails(self):
        result = run(RUSTFLAGS="-C target-cpu=znver5")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RUSTFLAGS", result.stdout)

    def test_ci_runs_gate(self):
        self.assertIn("python3 scripts/check-isa-rules.py", commands_for("ci-config"))
        policy = classify_paths(["scripts/check-isa-rules.py"])
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertFalse(policy.run_gpu)


if __name__ == "__main__":
    unittest.main()
