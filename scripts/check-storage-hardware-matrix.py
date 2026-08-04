#!/usr/bin/env python3
"""Run and record the frozen storage-provider hardware matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FREEZE = Path("docs/design/storage-contract-freeze.md")
LANES = {
    "cpu": {
        "argv": ["cargo", "test", "-p", "tenferro-tensor", "--test", "storage_public_api"],
        "evidence": "crates/tenferro-tensor/tests/storage_public_api.rs",
        "device": "host CPU (see environment)",
    },
    "cuda2": {
        "argv": [
            "cargo", "test", "-p", "tenferro-gpu", "--features", "cuda",
            "--test", "storage_provider_cuda", "--", "--nocapture",
        ],
        "evidence": "crates/tenferro-gpu/tests/storage_provider_cuda.rs",
        "device": "NVIDIA CUDA device(s), queried by the provider test",
    },
    "webgpu": {
        "argv": [
            "cargo", "test", "-p", "tenferro-gpu", "--features", "webgpu",
            "--test", "storage_provider_webgpu", "--", "--nocapture",
        ],
        "evidence": "crates/tenferro-gpu/tests/storage_provider_webgpu.rs",
        "device": "wgpu adapter, queried by WebGpuRuntime::new_default",
    },
    "metal": {
        "argv": [
            "cargo", "test", "-p", "tenferro-gpu", "--features", "webgpu",
            "--test", "integration", "--", "apple", "--nocapture",
        ],
        "evidence": "crates/tenferro-gpu/tests/integration/apple_context.rs",
        "device": "Apple Metal device, required on macOS",
    },
    "cuda-ad": {
        "argv": [
            "cargo", "test", "-p", "tenferro-ad", "--features", "cuda",
            "--test", "integration", "--", "gpu_ad_tests", "--nocapture",
        ],
        "evidence": "crates/tenferro-ad/tests/integration/gpu_ad_tests.rs",
        "device": "NVIDIA CUDA device used by AD integration tests",
    },
}
REQUIRED = ("cpu", "cuda2", "webgpu", "metal", "cuda-ad")


class CheckError(ValueError):
    pass


def read_freeze(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        raise CheckError("freeze report has no fenced JSON record")
    record = json.loads(match.group(1))
    candidate = record.get("candidate_commit")
    if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{40}", candidate):
        raise CheckError("freeze report has no full candidate commit")
    if record.get("status") != "pass":
        raise CheckError("freeze report is not a passing candidate")
    return record


def cpu_facts() -> str:
    model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[-1].strip()
                break
    return f"{platform.machine()} {model}; {platform.system()} {platform.release()}"


def run_lane(name: str, spec: dict, timeout: int) -> dict:
    start = time.monotonic()
    try:
        result = subprocess.run(
            spec["argv"], cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "lane": name,
            "status": "fail",
            "command": " ".join(spec["argv"]),
            "environment": cpu_facts(),
            "device_facts": spec["device"],
            "test_count": None,
            "duration_seconds": round(time.monotonic() - start, 3),
            "output_tail": str(error)[-2000:],
            "evidence": spec["evidence"],
            "skip_reason": None,
        }
    output = result.stdout
    counts = re.findall(r"test result: .*?(\d+) passed; (\d+) failed; (\d+) ignored", output)
    passed = sum(int(item[0]) for item in counts)
    failed = sum(int(item[1]) for item in counts)
    ignored = sum(int(item[2]) for item in counts)
    test_count = passed + failed + ignored if counts else None
    if result.returncode != 0:
        status = "fail"
        skip_reason = None
    elif test_count == 0:
        status = "skip"
        skip_reason = "no tests ran for this platform or no provider device was available"
    else:
        status = "pass"
        skip_reason = None
    return {
        "lane": name,
        "status": status,
        "command": " ".join(spec["argv"]),
        "environment": cpu_facts(),
        "device_facts": spec["device"],
        "test_count": test_count,
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "duration_seconds": round(time.monotonic() - start, 3),
        "output_tail": output[-2000:],
        "evidence": spec["evidence"],
        "skip_reason": skip_reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--required-mode", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--lanes", default=",".join(REQUIRED))
    args = parser.parse_args(argv)
    try:
        freeze = read_freeze(ROOT / FREEZE)
        lanes = [name for name in args.lanes.split(",") if name]
        unknown = [name for name in lanes if name not in LANES]
        if unknown:
            raise CheckError("unknown lane(s): " + ", ".join(unknown))
        results = [run_lane(name, LANES[name], args.timeout) for name in lanes]
        required_mode = args.required_mode or os.environ.get("TENFERRO_STORAGE_REQUIRED_HARDWARE") == "1"
        bad = [item for item in results if item["status"] == "fail"]
        skipped_required = [item for item in results if item["lane"] in REQUIRED and item["status"] == "skip"]
        if required_mode and skipped_required:
            raise CheckError("required hardware lane skipped: " + ", ".join(item["lane"] for item in skipped_required))
        if bad:
            raise CheckError("hardware lane failed: " + ", ".join(item["lane"] for item in bad))
        status = "pass" if not skipped_required else "structured-skip"
        report = args.report if not args.report.is_absolute() else args.report.relative_to(ROOT)
        if any(part == ".." for part in report.parts):
            raise CheckError("report path must remain inside the repository")
        record = {
            "schema": "tenferro.storage-hardware-matrix.v1",
            "candidate_commit": freeze["candidate_commit"],
            "required_lanes": list(REQUIRED),
            "required_mode": required_mode,
            "status": status,
            "environment": {"host": cpu_facts(), "python": platform.python_version()},
            "lanes": results,
            "evidence_paths": sorted({item["evidence"] for item in results}),
        }
        output = ROOT / report
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "# Frozen storage hardware matrix\n\n"
            "Unavailable hardware is a structured skip, never a pass.\n\n"
            "```json\n" + json.dumps(record, indent=2) + "\n```\n",
            encoding="utf-8",
        )
    except (CheckError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"storage-hardware-matrix: {error}", file=sys.stderr)
        return 1
    print(f"storage-hardware-matrix-{status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
