"""Exercise actual runtime seeding against a toolkit with alias directories."""
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "seed_cuda_runtime_tree.sh"

class RuntimeSeedTests(unittest.TestCase):
    def test_headers_shared_libraries_and_sonames_survive_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cuda"
            include = source / "targets/x86_64-linux/include"
            lib = source / "targets/x86_64-linux/lib"
            include.mkdir(parents=True)
            lib.mkdir()
            (source / "include").symlink_to("targets/x86_64-linux/include")
            (source / "lib64").symlink_to("targets/x86_64-linux/lib")
            (include / "cuda.h").write_text("header")
            (lib / "libnvrtc.so.12.8").write_bytes(b"runtime")
            (lib / "libnvrtc.so.12").symlink_to("libnvrtc.so.12.8")
            (lib / "libnvrtc.so").symlink_to("libnvrtc.so.12")
            (lib / "libunused_static.a").write_bytes(b"static")
            (lib / "stubs").mkdir()
            (lib / "stubs/libcuda.so").write_bytes(b"stub")
            (source / "bin").mkdir()
            (source / "bin/nvcc").write_bytes(b"compiler")
            destination = root / "runtime"
            subprocess.run(["bash", str(SCRIPT), str(source), str(destination)], check=True)
            self.assertEqual((destination / "include/cuda.h").read_text(), "header")
            self.assertEqual((destination / "lib64/libnvrtc.so").read_bytes(), b"runtime")
            self.assertTrue((destination / "lib64").is_symlink())
            self.assertTrue((destination / "lib64/libnvrtc.so.12").is_symlink())
            self.assertTrue((destination / ".seed-complete").exists())
            regular = [p for p in destination.rglob("*") if p.is_file() and not p.is_symlink()]
            self.assertEqual(sorted(p.name for p in regular), [".seed-complete", "cuda.h", "libnvrtc.so.12.8"])

    def test_broken_soname_does_not_mark_tree_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cuda"
            (source / "include").mkdir(parents=True)
            (source / "lib64").mkdir()
            (source / "lib64/libnvrtc.so").symlink_to("missing.so.12")
            destination = root / "runtime"
            result = subprocess.run(["bash", str(SCRIPT), str(source), str(destination)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((destination / ".seed-complete").exists())
