"""Execute the trusted workflow transport with real GNU tar/split/checksums."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_runpod_cost_contracts import CHILD, PARENT, step_script, text


class ArchiveTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "tenferro-rs"
        self.source.mkdir()
        self.env = dict(os.environ, CUDA_ARCHIVE="cuda-tests.tar.zst",
                        PJRT_ARCHIVE="pjrt-tests.tar.zst",
                        CUDA_TUTORIAL_ARCHIVE="cuda-tutorial.tar.zst",
                        GITHUB_STEP_SUMMARY=str(self.root / "summary"))
        # Unequal sizes make part boundaries cross original file boundaries.
        self.originals = {self.env[key]: os.urandom(size) for key, size in (
            ("CUDA_ARCHIVE", 190003), ("PJRT_ARCHIVE", 5001),
            ("CUDA_TUTORIAL_ARCHIVE", 7109))}
        for name, content in self.originals.items():
            (self.source / name).write_bytes(content)

    def run_step(self, workflow: str, name: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", step_script(text(workflow), name)],
                              cwd=self.root, env=self.env, capture_output=True, text=True)

    def prepare_download(self) -> None:
        prepared = self.run_step(PARENT, "Prepare five archive transfer parts")
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        transfer = self.root / "archive-transfer"
        sizes = []
        for part in range(5):
            directory = transfer / f"{part:02d}"
            path = directory / f"archives.part{part:02d}"
            sizes.append(path.stat().st_size)
            for item in directory.iterdir():
                item.rename(transfer / item.name)
            directory.rmdir()
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        (self.root / "archive-transfer-start").write_text("0\n")
        for name in self.originals:
            (self.source / name).unlink()

    def test_restores_identical_archives_and_records_elapsed_time(self) -> None:
        self.prepare_download()
        result = self.run_step(CHILD, "Reconstruct and verify archive transfer")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name, content in self.originals.items():
            self.assertEqual((self.source / name).read_bytes(), content)
        self.assertIn("reconstruction and verification:", (self.root / "summary").read_text())
        self.assertFalse((self.root / "archive-transfer").exists())

    def test_missing_part_fails_before_extracting(self) -> None:
        self.prepare_download()
        (self.root / "archive-transfer/archives.part03").unlink()
        result = self.run_step(CHILD, "Reconstruct and verify archive transfer")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.source.iterdir()), [])

    def test_corrupt_content_fails_verification(self) -> None:
        self.prepare_download()
        part = self.root / "archive-transfer/archives.part00"
        content = bytearray(part.read_bytes())
        # First tar header is 512 bytes; change archive content, not metadata.
        content[1000] ^= 1
        part.write_bytes(content)
        result = self.run_step(CHILD, "Reconstruct and verify archive transfer")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAILED", result.stdout)

    def test_five_named_uploads_and_parallel_download_retry(self) -> None:
        parent, child = text(PARENT), text(CHILD)
        for part in range(5):
            self.assertIn(f"artifact_name }}}}-transfer-part{part:02d}", parent)
        self.assertEqual(parent.split("  gpu-runtime:", 1)[0].count("compression-level: 0"), 5)
        self.assertEqual(child.count("pattern: ${{ inputs.archive_artifact_name }}-transfer-part*"), 2)
        self.assertEqual(child.count("pattern: ${{ inputs.archive_artifact_name }}-transfer-part*"), 2)
