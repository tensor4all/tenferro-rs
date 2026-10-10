"""Exercise the runtime-only cuTENSOR installer with real tar archives."""

import io
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERSION = "2.6.0.4"
PREFIX = f"libcutensor-linux-x86_64-{VERSION}_cuda12-archive/lib"


class CutensorRuntimeInstallTests(unittest.TestCase):
    def install(self, directory, *, shared=b"runtime library", link="libcutensor.so.2.6.0"):
        root = Path(directory)
        archive_path = root / "vendor.tar.xz"
        with tarfile.open(archive_path, "w:xz") as archive:
            members = {
                "libcutensor_static.a": b"unused static library",
                "libcutensorMg.so.2": b"unused multi-GPU library",
                "libcutensorMp.so.2": b"unused MPI library",
            }
            if shared is not None:
                members["libcutensor.so.2.6.0"] = shared
            for name, data in members.items():
                member = tarfile.TarInfo(f"{PREFIX}/{name}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            if link is not None:
                member = tarfile.TarInfo(f"{PREFIX}/libcutensor.so.2")
                member.type = tarfile.SYMTYPE
                member.linkname = link
                archive.addfile(member)
        binaries = root / "bin"
        binaries.mkdir()
        # Replace network/privilege operations and the large-download size
        # check only. Extraction and the final runtime ABI check stay real.
        for name, body in {
            "curl": 'while [ "$1" != -o ]; do shift; done\ncp "$FIXTURE_ARCHIVE" "$2"',
            "id": "echo 0",
            "wc": "cat >/dev/null\necho 400000000",
        }.items():
            executable = binaries / name
            executable.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
            executable.chmod(0o755)
        destination = root / "installed"
        destination.mkdir()
        (destination / "stale-static.a").write_text("old partial install")
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/ci/install_cutensor.sh"), VERSION, str(destination)],
            env=dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}",
                     FIXTURE_ARCHIVE=str(archive_path), TMPDIR=str(root)),
            capture_output=True, text=True,
        )
        return result, destination

    def test_installs_only_runtime_family_and_preserves_soname_link(self):
        with tempfile.TemporaryDirectory() as directory:
            result, destination = self.install(directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(sorted(p.name for p in destination.iterdir()), ["lib"])
            self.assertEqual(sorted(p.name for p in (destination / "lib").iterdir()),
                             ["libcutensor.so.2", "libcutensor.so.2.6.0"])
            self.assertEqual((destination / "lib/libcutensor.so.2").read_bytes(), b"runtime library")
            self.assertEqual(os.readlink(destination / "lib/libcutensor.so.2"), "libcutensor.so.2.6.0")

    def test_rejects_missing_empty_or_broken_runtime_abi(self):
        for shared, link in [(None, None), (b"", "libcutensor.so.2.6.0"),
                             (b"runtime library", "missing.so"), (b"runtime library", None)]:
            with self.subTest(shared=shared, link=link), tempfile.TemporaryDirectory() as directory:
                result, _ = self.install(directory, shared=shared, link=link)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
