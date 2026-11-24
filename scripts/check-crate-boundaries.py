#!/usr/bin/env python3
"""Check crate-boundary rules that are easy to regress mechanically."""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

from boundary_sources import boundary_lines


ROOT = pathlib.Path(__file__).resolve().parents[1]

# Operation families outside the core (#2694). They take the `autodiff` and `cuda`
# capabilities from the public crates, never from an internal one.
EXTENSION_CRATES = (
    "crates/tenferro-einsum",
    "crates/tenferro-fft",
    "crates/tenferro-linalg",
    "ext/sparse",
    "ext/tropical",
)
INTERNAL_PACKAGE = re.compile(r"^tenferro-internal-|^tenferro-cpu-basic$")


def iter_files(path: pathlib.Path):
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file():
            yield candidate


def dependency_packages(manifest: dict) -> dict[str, str]:
    """Map every dependency alias to its package name, including target tables."""
    packages: dict[str, str] = {}
    for key, value in manifest.items():
        if key in ("dependencies", "dev-dependencies", "build-dependencies"):
            for alias, spec in value.items():
                packages[alias] = spec.get("package", alias) if isinstance(spec, dict) else alias
        elif key == "target" and isinstance(value, dict):
            for target in value.values():
                packages.update(dependency_packages(target))
    return packages


def check_extension_features_use_public_crates() -> list[str]:
    """Extension crates forward a feature of a public crate, never of an internal one."""
    violations: list[str] = []
    for crate in EXTENSION_CRATES:
        manifest = tomllib.loads((ROOT / crate / "Cargo.toml").read_text(encoding="utf-8"))
        packages = dependency_packages(manifest)
        for feature, items in manifest.get("features", {}).items():
            for item in items:
                if item.startswith("dep:") or "/" not in item:
                    continue
                alias, _, _ = item.partition("/")
                # Dependency aliases are local names: resolve before the ban.
                package = packages.get(alias.removesuffix("?"), alias.removesuffix("?"))
                if INTERNAL_PACKAGE.match(package):
                    violations.append(
                        f"{crate}/Cargo.toml: feature `{feature}` takes `{item}` from the "
                        f"internal package `{package}`; take it from a public crate (#2694)"
                    )
    return violations


def check_tensor_has_no_gpu_runtime_deps() -> list[str]:
    forbidden = (
        "cubecl",
        "cudarc",
        "tenferro-gpu",
        "tenferro-internal-gpubackend",
    )
    tensor = ROOT / "crates" / "tenferro-tensor"
    paths = [tensor / "Cargo.toml", tensor / "src"]
    violations: list[str] = []
    for path in paths:
        for file_path in iter_files(path):
            for line_no, line in boundary_lines(file_path):
                for token in forbidden:
                    if re.search(r"\b" + re.escape(token).replace(r"\-", "[-_]") + r"\b", line):
                        rel = file_path.relative_to(ROOT)
                        violations.append(f"{rel}:{line_no}: forbidden tensor-core GPU dependency/reference `{token}`")
    return violations


def check_runtime_has_no_gpu_backend_deps() -> list[str]:
    forbidden = ("tenferro-gpu", "tenferro_gpu")
    runtime = ROOT / "crates" / "tenferro-runtime"
    paths = [runtime / "Cargo.toml", runtime / "src"]
    violations: list[str] = []
    for path in paths:
        for file_path in iter_files(path):
            for line_no, line in boundary_lines(file_path):
                for token in forbidden:
                    if re.search(r"\b" + re.escape(token) + r"\b", line):
                        rel = file_path.relative_to(ROOT)
                        violations.append(
                            f"{rel}:{line_no}: forbidden runtime GPU backend dependency/reference `{token}`"
                        )
    return violations


def check_internal_device_decomposed() -> list[str]:
    """Ensure the old catch-all device crate does not re-enter live crates."""

    violations: list[str] = []
    manifest = ROOT / "crates" / "tenferro-internal-device" / "Cargo.toml"
    if manifest.exists():
        violations.append(
            "crates/tenferro-internal-device/Cargo.toml: old device facade crate must be decomposed"
        )
    root_manifest = ROOT / "tenferro-internal-device" / "Cargo.toml"
    if root_manifest.exists():
        violations.append(
            "tenferro-internal-device/Cargo.toml: old root-level device facade crate must be decomposed"
        )

    forbidden = ("tenferro-internal-device", "tenferro_device")
    paths = [
        ROOT / ".github" / "workflows",
        ROOT / "crates" / "tenferro-einsum" / "Cargo.toml",
        ROOT / "crates" / "tenferro-einsum" / "src",
    ]
    for path in paths:
        for file_path in iter_files(path):
            text = file_path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                for token in forbidden:
                    if token in line:
                        rel = file_path.relative_to(ROOT)
                        violations.append(
                            f"{rel}:{line_no}: forbidden old device crate token `{token}`"
                        )
    return violations


def main() -> int:
    violations = [
        *check_tensor_has_no_gpu_runtime_deps(),
        *check_runtime_has_no_gpu_backend_deps(),
        *check_internal_device_decomposed(),
        *check_extension_features_use_public_crates(),
    ]
    if violations:
        print("crate-boundary check failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("crate-boundary-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
