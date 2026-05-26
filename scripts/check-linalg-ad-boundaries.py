#!/usr/bin/env python3
"""Check linalg AD packaging and feature-boundary rules."""

from __future__ import annotations

from pathlib import Path
import sys


SKIP_DIRS = {
    ".git",
    "target",
    "docs/plans",
    "docs/superpowers",
}


def is_skipped(repo: Path, path: Path) -> bool:
    rel = path.relative_to(repo).as_posix()
    return any(rel == skip or rel.startswith(skip + "/") for skip in SKIP_DIRS)


def iter_text_files(repo: Path):
    for path in repo.rglob("*"):
        if is_skipped(repo, path) or not path.is_file():
            continue
        if path.suffix in {".toml", ".md", ".rs", ".py"}:
            yield path


def has_line(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    findings: list[str] = []

    removed_crate = repo / ("tenferro-linalg" + "-ad") / "Cargo.toml"
    if removed_crate.exists():
        findings.append(
            "forbidden standalone linalg AD crate exists: " + "tenferro-linalg" + "-ad"
        )

    linalg_manifest = repo / "tenferro-linalg" / "Cargo.toml"
    if not has_line(linalg_manifest, "autodiff = ["):
        findings.append("tenferro-linalg/Cargo.toml is missing the autodiff feature")
    optional_ad_dep = (
        'tenferro-ad = { path = "../tenferro-ad", default-features = false, optional = true }'
    )
    if not has_line(linalg_manifest, optional_ad_dep):
        findings.append("tenferro-linalg must keep tenferro-ad optional")

    linalg_lib = repo / "tenferro-linalg" / "src" / "lib.rs"
    for needle in [
        '#[cfg(feature = "autodiff")]\nmod ad;',
        '#[cfg(feature = "autodiff")]\npub mod eager_tensor;',
        '#[cfg(feature = "autodiff")]\npub use ad::{ad_rules, register_extension_rule};',
    ]:
        if needle not in linalg_lib.read_text(encoding="utf-8"):
            findings.append(f"tenferro-linalg/src/lib.rs missing gated item: {needle!r}")

    operation_manifests = [
        repo / "tenferro-einsum" / "Cargo.toml",
        repo / "tenferro-fft" / "Cargo.toml",
        repo / "tenferro-linalg" / "Cargo.toml",
    ]
    for manifest in operation_manifests:
        text = manifest.read_text(encoding="utf-8")
        if "autodiff = [" not in text:
            findings.append(f"{manifest.relative_to(repo)} is missing the public autodiff feature")
        if "\nad =" in text:
            findings.append(f"{manifest.relative_to(repo)} must not expose an ad feature alias")
        for feature in ["cuda = [", "rocm = ["]:
            if feature not in text:
                findings.append(f"{manifest.relative_to(repo)} is missing the {feature[:-4]} feature")

    public_manifests = [repo / "tenferro-ad" / "Cargo.toml", *operation_manifests]
    for manifest in public_manifests:
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if line.startswith("gpu ="):
                findings.append(
                    f"{manifest.relative_to(repo)}:{line_no}: public gpu feature is forbidden"
                )
            if "tenferro-ad/gpu" in line:
                findings.append(
                    f"{manifest.relative_to(repo)}:{line_no}: use cuda/rocm, not tenferro-ad/gpu"
                )

    forbidden_refs = ["tenferro-linalg" + "-ad", "tenferro_linalg" + "_ad"]
    for path in iter_text_files(repo):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden_refs:
            if token in text:
                findings.append(f"{path.relative_to(repo)} references removed crate token {token!r}")

    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
