import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "scripts" / "lib" / "python.sh"
VERSION_PROBE = (
    "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
)


class PythonResolverTests(unittest.TestCase):
    def run_resolver(self, path, *args, python=None, extra_env=None):
        env = {"PATH": str(path)}
        if python is not None:
            env["PYTHON"] = python
        if extra_env:
            env.update(extra_env)
        command = 'set -euo pipefail; source "$1"; shift; run_python "$@"'
        return subprocess.run(
            ["/bin/bash", "-c", command, "resolver", str(HELPER), *args],
            env=env,
            text=True,
            capture_output=True,
        )

    @staticmethod
    def fake_executable(directory, name, label, version_ok=True, log_path=None):
        path = Path(directory) / name
        log_expression = ""
        if log_path is not None:
            log_expression = f'printf \'%s\\n\' "$*" >> {str(log_path)!r}\n'
        probe_status = 0 if version_ok else 1
        path.write_text(
            "#!/bin/bash\n"
            "set -eu\n"
            f"{log_expression}"
            f'if [[ "${{1-}}" == "-c" && "${{2-}}" == {VERSION_PROBE!r} ]]; then\n'
            f"  exit {probe_status}\n"
            "fi\n"
            'if [[ "${1-}" == "--version" ]]; then\n'
            "  printf 'Python 3.11.0\\n'\n"
            "  exit 0\n"
            "fi\n"
            f"printf '%s\\n' {label!r}\n"
        )
        path.chmod(0o755)
        return path

    def test_old_python3_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.fake_executable(temporary, "python3", "old-python3", version_ok=False)
            result = self.run_resolver(temporary, "-c", "print('should not run')")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 3.11+", result.stderr)
        self.assertNotIn("should not run", result.stdout)

    def test_banner_spoofing_without_semantic_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            spoof = Path(temporary) / "python3.11"
            spoof.write_text(
                "#!/bin/bash\n"
                "set -eu\n"
                'if [[ "${1-}" == "--version" ]]; then\n'
                "  printf 'Python 3.11.0\\n'\n"
                "  exit 0\n"
                "fi\n"
                f'if [[ "${{1-}}" == "-c" && "${{2-}}" == {VERSION_PROBE!r} ]]; then\n'
                "  exit 1\n"
                "fi\n"
                "printf 'not-python\\n'\n"
            )
            spoof.chmod(0o755)
            result = self.run_resolver(temporary, "-c", "print('selected')")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 3.11+", result.stderr)
        self.assertNotIn("not-python", result.stdout)

    def test_python311_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.fake_executable(temporary, "python3.11", "python311")
            self.fake_executable(temporary, "python3", "python3")
            result = self.run_resolver(temporary, "-c", "print('selected')")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "python311\n")
        self.assertEqual(result.stderr, "")

    def test_python_override_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            override = self.fake_executable(temporary, "override-python", "override")
            self.fake_executable(temporary, "python3.13", "python313")
            result = self.run_resolver(
                temporary,
                "-c",
                "print('selected')",
                python=str(override),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "override\n")

    def test_invalid_python_override_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.fake_executable(temporary, "python3.11", "fallback")
            result = self.run_resolver(
                temporary,
                "-c",
                "print('selected')",
                python=str(Path(temporary) / "missing-python"),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("$PYTHON", result.stderr)
        self.assertNotIn("fallback", result.stdout)

    def test_uv_uses_exact_command_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "uv.log"
            uv = Path(temporary) / "uv"
            uv.write_text(
                "#!/bin/bash\n"
                "set -eu\n"
                f"printf '%s\\n' \"$*\" >> {str(log_path)!r}\n"
                '[[ "${1-}" == "run" ]]\n'
                '[[ "${2-}" == "--no-project" ]]\n'
                '[[ "${3-}" == "--python" ]]\n'
                '[[ "${4-}" == "3.12" ]]\n'
                '[[ "${5-}" == "python" ]]\n'
                f'if [[ "${{6-}}" == "-c" && "${{7-}}" == {VERSION_PROBE!r} ]]; then exit 0; fi\n'
                "printf 'uv-selected\\n'\n"
            )
            uv.chmod(0o755)
            result = self.run_resolver(temporary, "-c", "print('selected')")
            uv_calls = log_path.read_text().splitlines() if log_path.exists() else []

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "uv-selected\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(uv_calls), 1)
        self.assertTrue(uv_calls[0].startswith("run --no-project --python 3.12 python -c"))

    def test_no_candidate_error_documents_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_resolver(temporary, "-c", "print('selected')")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 3.11+", result.stderr)
        self.assertIn("$PYTHON", result.stderr)
        self.assertIn("uv", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_relative_override_is_resolved_from_source_directory_and_cached(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            source_dir = temp / "caller path with spaces"
            source_dir.mkdir()
            (source_dir / "bin").mkdir()
            first_dir = temp / "first dir"
            second_dir = temp / "second dir"
            first_dir.mkdir()
            second_dir.mkdir()
            empty_path = temp / "empty-path"
            empty_path.mkdir()
            log_path = temp / "python.log"
            self.fake_executable(
                source_dir / "bin",
                "python executable",
                "relative",
                log_path=log_path,
            )
            command = (
                'set -euo pipefail; source "$1"; shift; '
                'cd "$1"; run_python "${@:3}"; '
                'cd "$2"; run_python "${@:3}"'
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    "-c",
                    command,
                    "resolver",
                    str(HELPER),
                    str(first_dir),
                    str(second_dir),
                    "-c",
                    "print('selected')",
                ],
                cwd=source_dir,
                env={"PATH": str(empty_path), "PYTHON": "bin/python executable"},
                text=True,
                capture_output=True,
            )
            calls = log_path.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "relative\nrelative\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(calls, [f"-c {VERSION_PROBE}", "-c print('selected')", "-c print('selected')"])

    def test_selection_is_cached_and_probes_are_silent(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "python.log"
            self.fake_executable(temporary, "python3.11", "python311", log_path=log_path)
            command = (
                'set -euo pipefail; source "$1"; shift; '
                'run_python "$@"; run_python "$@"'
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    "-c",
                    command,
                    "resolver",
                    str(HELPER),
                    "-c",
                    "print('selected')",
                ],
                env={"PATH": temporary, "FAKE_LOG": str(log_path)},
                text=True,
                capture_output=True,
            )
            calls = log_path.read_text().splitlines() if log_path.exists() else []

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "python311\npython311\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], f"-c {VERSION_PROBE}")


if __name__ == "__main__":
    unittest.main()
