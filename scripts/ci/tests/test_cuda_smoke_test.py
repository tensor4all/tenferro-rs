import unittest

from scripts.ci.cuda_smoke_test import (
    EXPECTED_OUTPUT,
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
