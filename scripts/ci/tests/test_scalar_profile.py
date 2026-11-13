import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("scalar_check", ROOT / "scripts/check-scalar-composition-kernel-sharing.py")
scalar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scalar)


class ScalarProfileTests(unittest.TestCase):
    def test_assemblies_are_found_in_both_cargo_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "ci"
            old = profile / "deps" / "composition-aaaa.s"
            new = profile / "build" / "tenferro-df64-proof" / "bbbb" / "out" / "composition-bbbb.s"
            for path in (old, new):
                path.parent.mkdir(parents=True)
                path.write_text("")
            (profile / "deps" / "other-cccc.s").write_text("")
            found = set(scalar.emitted_assemblies(profile, "composition"))
            self.assertEqual(found, {old, new})

    def test_profile_controls_both_builds_and_assembly_lookup(self) -> None:
        for options, profile, directory in (([], "release", "release"), (["--debug"], "dev", "debug"), (["--profile", "ci"], "ci", "ci")):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                deps = root / "target" / directory / "deps"
                deps.mkdir(parents=True)
                assembly = ""
                for function in scalar.KERNEL_FUNCTIONS:
                    for member in (scalar.PRESET_OP, scalar.EXTERNAL_MEMBER + scalar.EXTERNAL_OP):
                        symbol = "_R" + scalar.SHARED_KERNEL_CRATE + function + member
                        assembly += f"{symbol}:\n\tcall {symbol}\n\tcall {symbol}\n"
                for name in ("composition", "comparison"):
                    (deps / f"{name}-test.s").write_text(assembly)
                report = root / "report.md"
                argv = ["scalar", "--report", str(report), "--against-package", "consumer", "--against-target", "comparison", *options]
                with patch.object(scalar, "ROOT", root), patch.object(scalar, "run", return_value="metadata"), patch.object(scalar.subprocess, "run") as run, patch("sys.argv", argv), patch.dict("os.environ", {}, clear=True), patch("builtins.print"):
                    self.assertEqual(scalar.main(), 0)
                self.assertEqual(run.call_count, 2)
                for call in run.call_args_list:
                    command = call.args[0]
                    self.assertEqual(command[command.index("--profile") + 1], profile)
                data = json.loads(report.read_text().split("```json\n")[1].split("\n```")[0])
                self.assertEqual(data["profile"], profile)
                self.assertEqual(data["status"], "pass")
                self.assertEqual(data["comparison"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
