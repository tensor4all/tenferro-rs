import hashlib
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from scripts.ci import cuda_smoke_test

from scripts.ci.cuda_smoke_test import (
    EXPECTED_OUTPUT,
    install_nvrtc,
    nvrtc_builtins_candidates,
    nvrtc_library_candidates,
    SmokeFailure,
    nvrtc_arch_option,
    nvrtc_package,
    parse_driver_cuda_version,
    parse_version,
    run_smoke,
    select_runtime_version,
)

SMI_OUTPUT = (
    "| NVIDIA-SMI 550.127.05    Driver Version: 550.127.05    "
    "CUDA Version: 12.4     |"
)


class InstallNvrtcTests(unittest.TestCase):
    def test_verified_package_installs_without_apt_and_temp_file_is_removed(self):
        content = b"test NVRTC package"
        digest = hashlib.sha256(content).hexdigest()
        archive_paths = []

        def execute(args, **kwargs):
            if args[0] == "curl":
                archive = Path(args[args.index("-o") + 1])
                archive.write_bytes(content)
                archive_paths.append(archive)
                self.assertTrue(args[-1].endswith("cuda-nvrtc-12-8_12.8.93-1_amd64.deb"))
            else:
                self.assertEqual(args[:2], ["dpkg", "-i"])
                self.assertEqual(Path(args[2]).read_bytes(), content)
            self.assertTrue(kwargs["check"])

        with patch.dict(cuda_smoke_test.NVRTC_PACKAGES, {(12, 8): ("12.8.93-1", digest)}), \
             patch.object(cuda_smoke_test.subprocess, "run", side_effect=execute) as run:
            install_nvrtc((12, 8))
        self.assertEqual(run.call_count, 2)
        self.assertFalse(archive_paths[0].exists())

    def test_corrupt_download_never_installs(self):
        def download(args, **kwargs):
            Path(args[args.index("-o") + 1]).write_bytes(b"incomplete download")

        with patch.object(cuda_smoke_test.subprocess, "run", side_effect=download) as run:
            with self.assertRaisesRegex(SmokeFailure, "checksum mismatch"):
                install_nvrtc((12, 8))
        self.assertEqual(run.call_count, 1)

    def test_download_failure_never_installs(self):
        with patch.object(cuda_smoke_test.subprocess, "run",
                          side_effect=subprocess.CalledProcessError(22, "curl")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                install_nvrtc((12, 8))
        self.assertEqual(run.call_count, 1)

    def test_unknown_runtime_fails_before_download(self):
        with patch.object(cuda_smoke_test.subprocess, "run") as run:
            with self.assertRaisesRegex(SmokeFailure, "no pinned NVRTC"):
                install_nvrtc((99, 0))
        run.assert_not_called()


class VersionLogicTests(unittest.TestCase):
    def test_parses_driver_cuda_version_from_nvidia_smi(self) -> None:
        self.assertEqual(parse_driver_cuda_version(SMI_OUTPUT), (12, 4))

    def test_missing_driver_version_is_a_failure(self) -> None:
        with self.assertRaises(SmokeFailure):
            parse_driver_cuda_version("no gpu here")

    def test_runtime_selection_mirrors_workflow_tiers(self) -> None:
        minimum, full = (12, 4), (12, 8)
        self.assertEqual(
            select_runtime_version((12, 4), minimum=minimum, full=full), (12, 4)
        )
        self.assertEqual(
            select_runtime_version((12, 8), minimum=minimum, full=full), (12, 8)
        )
        self.assertEqual(
            select_runtime_version((13, 0), minimum=minimum, full=full), (12, 8)
        )
        with self.assertRaises(SmokeFailure):
            select_runtime_version((12, 2), minimum=minimum, full=full)

    def test_nvrtc_load_order_prefers_selected_runtime_paths(self) -> None:
        """The image's older bare libnvrtc.so must never shadow the tier.

        Observed live: bare "libnvrtc.so" resolved to the pod image's NVRTC
        11.8 and failed the proof on every otherwise-compatible host.
        """

        candidates = nvrtc_library_candidates((12, 8))
        self.assertEqual(
            candidates,
            [
                "/usr/local/cuda-12.8/lib64/libnvrtc.so.12",
                "/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12",
                "libnvrtc.so.12",
            ],
        )
        self.assertNotIn("libnvrtc.so", candidates)

    def test_nvrtc_builtins_preload_order(self) -> None:
        """NVRTC dlopens builtins by soname; absolute paths must come first."""

        candidates = nvrtc_builtins_candidates((12, 8))
        self.assertEqual(
            candidates,
            [
                "/usr/local/cuda-12.8/lib64/libnvrtc-builtins.so.12.8",
                "/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc-builtins.so.12.8",
                "libnvrtc-builtins.so.12.8",
            ],
        )

    def test_package_and_arch_option_naming(self) -> None:
        self.assertEqual(nvrtc_package((12, 8)), "cuda-nvrtc-12-8")
        self.assertEqual(nvrtc_arch_option(8, 9), b"--gpu-architecture=compute_89")
        self.assertEqual(parse_version("12.8"), (12, 8))
        self.assertEqual(parse_version("13"), (13, 0))


class FakeBindings:
    def __init__(
        self,
        *,
        nvrtc=(12, 8),
        properties=(8, 9, 24 * 1024**3),
        ptx=b"fake-ptx",
        output=EXPECTED_OUTPUT,
    ) -> None:
        self._nvrtc = nvrtc
        self._properties = properties
        self._ptx = ptx
        self._output = output
        self.calls: list[str] = []

    def nvrtc_version(self):
        self.calls.append("nvrtc_version")
        return self._nvrtc

    def device_properties(self):
        self.calls.append("device_properties")
        return self._properties

    def compile_to_ptx(self, source, arch_option):
        self.calls.append(f"compile:{arch_option.decode()}")
        return self._ptx

    def launch_ptx(self, ptx):
        self.calls.append("launch")
        return self._output


class RunSmokeTests(unittest.TestCase):
    def test_full_proof_sequence_passes(self) -> None:
        bindings = FakeBindings()
        run_smoke(bindings, driver=(12, 8), runtime=(12, 8), min_vram_gb=16)
        self.assertEqual(
            bindings.calls,
            [
                "nvrtc_version",
                "device_properties",
                "compile:--gpu-architecture=compute_89",
                "launch",
            ],
        )

    def test_nvrtc_newer_than_driver_fails_before_compile(self) -> None:
        bindings = FakeBindings(nvrtc=(12, 8))
        with self.assertRaises(SmokeFailure):
            run_smoke(bindings, driver=(12, 4), runtime=(12, 4), min_vram_gb=0)
        self.assertNotIn("launch", bindings.calls)

    def test_nvrtc_older_than_runtime_fails(self) -> None:
        bindings = FakeBindings(nvrtc=(12, 4))
        with self.assertRaises(SmokeFailure):
            run_smoke(bindings, driver=(12, 8), runtime=(12, 8), min_vram_gb=0)

    def test_insufficient_vram_fails_before_compile(self) -> None:
        bindings = FakeBindings(properties=(8, 9, 8 * 1024**3))
        with self.assertRaises(SmokeFailure):
            run_smoke(bindings, driver=(12, 8), runtime=(12, 8), min_vram_gb=16)
        self.assertNotIn("launch", bindings.calls)

    def test_wrong_kernel_output_fails(self) -> None:
        bindings = FakeBindings(output=0)
        with self.assertRaises(SmokeFailure):
            run_smoke(bindings, driver=(12, 8), runtime=(12, 8), min_vram_gb=0)

    def test_empty_ptx_fails_before_launch(self) -> None:
        bindings = FakeBindings(ptx=b"")
        with self.assertRaises(SmokeFailure):
            run_smoke(bindings, driver=(12, 8), runtime=(12, 8), min_vram_gb=0)
        self.assertNotIn("launch", bindings.calls)


if __name__ == "__main__":
    unittest.main()
