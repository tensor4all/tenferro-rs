#!/usr/bin/env python3
"""Check that linalg AD support stays out of the primal linalg crate."""

from __future__ import annotations

from pathlib import Path
import re
import sys


PRIMAL_FORBIDDEN = re.compile(
    r"\b(ExtensionAdRule|autodiff|chainrules|computegraph|eager_tensor|tenferro-ad)\b"
)
REMOVED_FACADE_AD_FEATURE = "tenferro" + "/autodiff"


def scan_file(repo: Path, path: Path, pattern: re.Pattern[str]) -> list[str]:
    findings: list[str] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if pattern.search(line):
            findings.append(f"{path.relative_to(repo)}:{line_no}: {line}")
    return findings


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    findings: list[str] = []
    checked = [repo / "tenferro-linalg" / "Cargo.toml"]
    checked.extend((repo / "tenferro-linalg" / "src").rglob("*.rs"))
    checked.extend((repo / "tenferro-linalg" / "tests").rglob("*.rs"))
    for path in checked:
        findings.extend(scan_file(repo, path, PRIMAL_FORBIDDEN))

    for manifest in [
        repo / "tenferro-einsum" / "Cargo.toml",
        repo / "tenferro-fft" / "Cargo.toml",
        repo / "tenferro-linalg" / "Cargo.toml",
    ]:
        for line_no, line in enumerate(manifest.read_text().splitlines(), start=1):
            if REMOVED_FACADE_AD_FEATURE in line:
                findings.append(f"{manifest.relative_to(repo)}:{line_no}: {line}")

    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
