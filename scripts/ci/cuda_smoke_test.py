#!/usr/bin/env python3
"""Prove CUDA compatibility on a RunPod host before runner registration.

GPU model metadata alone does not establish CUDA/PTX compatibility (#1404):
observed same-SKU hosts carried different drivers, and only a runtime proof
distinguishes them. This script runs on the pod, as root, BEFORE the GitHub
runner registers, so an incompatible host is rejected before any dependency
setup or paid test time:

1. read the driver CUDA API version from ``nvidia-smi``,
2. select the CUDA runtime tier exactly like the test workflow does,
3. install only the matching NVRTC package,
4. compile a tiny kernel with NVRTC for the device's compute capability,
5. load the generated PTX through the driver, launch, synchronize, and
   verify the kernel's output,
6. check total VRAM against the configured requirement.

Exit code 0 means the host is proven compatible; any failure prints a
``SMOKE FAIL:`` line and exits nonzero, which stops the startup script
before ``run.sh --jitconfig`` so the trusted provision loop deletes the pod
and retries the next candidate. This script must never receive or read any
credential.
"""

from __future__ import annotations

import argparse
import ctypes
import re
import subprocess
import sys

SMOKE_KERNEL = (
    'extern "C" __global__ void tenferro_smoke(int *out) { out[0] = 42; }'
)
KERNEL_NAME = b"tenferro_smoke"
EXPECTED_OUTPUT = 42


class SmokeFailure(RuntimeError):
    """The host failed a compatibility proof step."""


def parse_driver_cuda_version(nvidia_smi_output: str) -> tuple[int, int]:
    """Extract the driver CUDA API version from nvidia-smi output."""

    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", nvidia_smi_output)
    if match is None:
        raise SmokeFailure("driver CUDA API version not found in nvidia-smi output")
    return int(match.group(1)), int(match.group(2))


def parse_version(text: str) -> tuple[int, int]:
    major, _, minor = text.partition(".")
    return int(major), int(minor or 0)


def select_runtime_version(
    driver: tuple[int, int],
    *,
    minimum: tuple[int, int],
    full: tuple[int, int],
) -> tuple[int, int]:
    """Mirror the workflow's driver-based CUDA runtime tier selection."""

    if driver < minimum:
        raise SmokeFailure(
            f"driver CUDA API {driver[0]}.{driver[1]} is below the "
            f"{minimum[0]}.{minimum[1]} minimum"
        )
    return full if driver >= full else minimum


def nvrtc_package(runtime: tuple[int, int]) -> str:
    return f"cuda-nvrtc-{runtime[0]}-{runtime[1]}"


def nvrtc_arch_option(cc_major: int, cc_minor: int) -> bytes:
    return f"--gpu-architecture=compute_{cc_major}{cc_minor}".encode()


def install_nvrtc(runtime: tuple[int, int]) -> None:
    """Install only the NVRTC package for the selected runtime tier."""

    keyring_check = subprocess.run(
        "ls /etc/apt/sources.list.d/cuda*.list",
        shell=True,
        capture_output=True,
    )
    if keyring_check.returncode != 0:
        subprocess.run(
            "tmpdir=$(mktemp -d) && "
            "curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors "
            "-o \"${tmpdir}/cuda-keyring.deb\" "
            "https://developer.download.nvidia.com/compute/cuda/repos/"
            "ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && "
            "dpkg -i \"${tmpdir}/cuda-keyring.deb\" && rm -rf \"${tmpdir}\"",
            shell=True,
            check=True,
        )
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            nvrtc_package(runtime),
        ],
        check=True,
        env={"DEBIAN_FRONTEND": "noninteractive", "PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
    )


def _load_library(names: list[str]) -> ctypes.CDLL:
    last_error: OSError | None = None
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as error:
            last_error = error
    raise SmokeFailure(f"none of {names} could be loaded: {last_error}")


class CudaBindings:
    """Thin ctypes wrapper; tests substitute a fake with the same surface."""

    def __init__(self, runtime: tuple[int, int]) -> None:
        major = runtime[0]
        self.nvrtc = _load_library(
            [
                f"libnvrtc.so.{major}",
                "libnvrtc.so",
                f"/usr/local/cuda-{runtime[0]}.{runtime[1]}/lib64/libnvrtc.so.{major}",
                f"/usr/local/cuda-{runtime[0]}.{runtime[1]}"
                f"/targets/x86_64-linux/lib/libnvrtc.so.{major}",
            ]
        )
        self.cuda = _load_library(["libcuda.so.1", "libcuda.so"])

    def nvrtc_version(self) -> tuple[int, int]:
        major = ctypes.c_int()
        minor = ctypes.c_int()
        status = self.nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        if status != 0:
            raise SmokeFailure(f"nvrtcVersion failed with status {status}")
        return major.value, minor.value

    def compile_to_ptx(self, source: str, arch_option: bytes) -> bytes:
        program = ctypes.c_void_p()
        status = self.nvrtc.nvrtcCreateProgram(
            ctypes.byref(program),
            source.encode(),
            b"tenferro_smoke.cu",
            0,
            None,
            None,
        )
        if status != 0:
            raise SmokeFailure(f"nvrtcCreateProgram failed with status {status}")
        options = (ctypes.c_char_p * 1)(arch_option)
        status = self.nvrtc.nvrtcCompileProgram(program, 1, options)
        if status != 0:
            log_size = ctypes.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
            log = ctypes.create_string_buffer(log_size.value)
            self.nvrtc.nvrtcGetProgramLog(program, log)
            raise SmokeFailure(
                f"NVRTC compilation failed with status {status}: "
                f"{log.value.decode(errors='replace')}"
            )
        ptx_size = ctypes.c_size_t()
        self.nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
        ptx = ctypes.create_string_buffer(ptx_size.value)
        status = self.nvrtc.nvrtcGetPTX(program, ptx)
        if status != 0:
            raise SmokeFailure(f"nvrtcGetPTX failed with status {status}")
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
        return ptx.value

    def _check(self, name: str, status: int) -> None:
        if status != 0:
            raise SmokeFailure(f"{name} failed with CUDA error {status}")

    def device_properties(self) -> tuple[int, int, int]:
        """Return (cc_major, cc_minor, total_memory_bytes) for device 0."""

        self._check("cuInit", self.cuda.cuInit(0))
        device = ctypes.c_int()
        self._check("cuDeviceGet", self.cuda.cuDeviceGet(ctypes.byref(device), 0))
        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        # 75/76: CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR
        self._check(
            "cuDeviceGetAttribute(cc_major)",
            self.cuda.cuDeviceGetAttribute(ctypes.byref(cc_major), 75, device),
        )
        self._check(
            "cuDeviceGetAttribute(cc_minor)",
            self.cuda.cuDeviceGetAttribute(ctypes.byref(cc_minor), 76, device),
        )
        total_mem = ctypes.c_size_t()
        self._check(
            "cuDeviceTotalMem",
            self.cuda.cuDeviceTotalMem_v2(ctypes.byref(total_mem), device),
        )
        return cc_major.value, cc_minor.value, total_mem.value

    def launch_ptx(self, ptx: bytes) -> int:
        """Load PTX, launch the smoke kernel, synchronize, return out[0]."""

        device = ctypes.c_int()
        self._check("cuDeviceGet", self.cuda.cuDeviceGet(ctypes.byref(device), 0))
        context = ctypes.c_void_p()
        self._check(
            "cuCtxCreate",
            self.cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device),
        )
        try:
            module = ctypes.c_void_p()
            self._check(
                "cuModuleLoadData",
                self.cuda.cuModuleLoadData(ctypes.byref(module), ptx),
            )
            function = ctypes.c_void_p()
            self._check(
                "cuModuleGetFunction",
                self.cuda.cuModuleGetFunction(
                    ctypes.byref(function), module, KERNEL_NAME
                ),
            )
            device_ptr = ctypes.c_ulonglong()
            self._check(
                "cuMemAlloc",
                self.cuda.cuMemAlloc_v2(ctypes.byref(device_ptr), 4),
            )
            params = (ctypes.c_void_p * 1)(
                ctypes.cast(ctypes.byref(device_ptr), ctypes.c_void_p)
            )
            self._check(
                "cuLaunchKernel",
                self.cuda.cuLaunchKernel(
                    function, 1, 1, 1, 1, 1, 1, 0, None, params, None
                ),
            )
            self._check("cuCtxSynchronize", self.cuda.cuCtxSynchronize())
            result = ctypes.c_int()
            self._check(
                "cuMemcpyDtoH",
                self.cuda.cuMemcpyDtoH_v2(
                    ctypes.byref(result), device_ptr, 4
                ),
            )
            self.cuda.cuMemFree_v2(device_ptr)
            return result.value
        finally:
            self.cuda.cuCtxDestroy_v2(context)


def run_smoke(
    bindings: object,
    *,
    driver: tuple[int, int],
    runtime: tuple[int, int],
    min_vram_gb: float,
) -> None:
    """Run the compatibility proof against loaded driver/NVRTC bindings."""

    nvrtc_major, nvrtc_minor = bindings.nvrtc_version()
    print(f"Loaded NVRTC version: {nvrtc_major}.{nvrtc_minor}")
    if (nvrtc_major, nvrtc_minor) < runtime:
        raise SmokeFailure(
            f"loaded NVRTC {nvrtc_major}.{nvrtc_minor} is older than the "
            f"selected runtime {runtime[0]}.{runtime[1]}"
        )
    if (nvrtc_major, nvrtc_minor) > driver:
        raise SmokeFailure(
            f"loaded NVRTC {nvrtc_major}.{nvrtc_minor} is newer than the "
            f"driver CUDA API {driver[0]}.{driver[1]}"
        )

    cc_major, cc_minor, total_mem = bindings.device_properties()
    vram_gb = total_mem / (1024**3)
    print(f"Device compute capability: {cc_major}.{cc_minor}")
    print(f"Device total VRAM: {vram_gb:.1f} GB")
    if vram_gb < min_vram_gb:
        raise SmokeFailure(
            f"device VRAM {vram_gb:.1f} GB is below the required "
            f"{min_vram_gb:g} GB"
        )

    ptx = bindings.compile_to_ptx(
        SMOKE_KERNEL, nvrtc_arch_option(cc_major, cc_minor)
    )
    if not ptx:
        raise SmokeFailure("NVRTC produced empty PTX")
    print(f"NVRTC compiled smoke kernel to {len(ptx)} bytes of PTX.")

    output = bindings.launch_ptx(ptx)
    if output != EXPECTED_OUTPUT:
        raise SmokeFailure(
            f"smoke kernel returned {output}, expected {EXPECTED_OUTPUT}"
        )
    print("PTX load, kernel launch, synchronize, and readback all succeeded.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-runtime-version", default="12.4")
    parser.add_argument("--full-runtime-version", default="12.8")
    parser.add_argument("--min-vram-gb", type=float, default=0.0)
    parser.add_argument(
        "--skip-nvrtc-install",
        action="store_true",
        help="Assume a matching NVRTC is already present (for reruns)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        if smi.returncode != 0:
            raise SmokeFailure(
                f"nvidia-smi failed with status {smi.returncode}"
            )
        driver = parse_driver_cuda_version(smi.stdout)
        print(f"Driver CUDA API: {driver[0]}.{driver[1]}")
        runtime = select_runtime_version(
            driver,
            minimum=parse_version(args.min_runtime_version),
            full=parse_version(args.full_runtime_version),
        )
        print(f"Selected CUDA runtime tier: {runtime[0]}.{runtime[1]}")
        if not args.skip_nvrtc_install:
            install_nvrtc(runtime)
        bindings = CudaBindings(runtime)
        run_smoke(
            bindings,
            driver=driver,
            runtime=runtime,
            min_vram_gb=args.min_vram_gb,
        )
    except (SmokeFailure, subprocess.CalledProcessError, OSError) as error:
        print(f"SMOKE FAIL: {error}", flush=True)
        return 1
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
