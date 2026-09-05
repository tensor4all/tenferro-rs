#!/usr/bin/env python3
"""Classify a diff into conservative CI lanes."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import os
import subprocess
from pathlib import Path
from typing import Sequence


class ChangeClass(enum.StrEnum):
    """Primary change class used by local and hosted CI."""

    CODE = "code"
    DOCS_ONLY = "docs-only"
    CI_ONLY = "ci-only"


@dataclasses.dataclass(frozen=True)
class ChangePolicy:
    """Exact CI lanes selected for a set of changed paths."""

    change_class: ChangeClass
    run_rust: bool
    run_blas: bool
    run_extensions: bool
    run_docs: bool
    run_ci_config: bool
    run_gpu: bool
    run_macos: bool
    reasons: tuple[str, ...]

    def as_output(self) -> dict[str, str | bool]:
        """Return stable names consumed by GitHub Actions."""

        return {
            "classification": self.change_class.value,
            "run_rust": self.run_rust,
            "run_blas": self.run_blas,
            "run_extensions": self.run_extensions,
            "run_docs": self.run_docs,
            "run_ci_config": self.run_ci_config,
            "run_gpu": self.run_gpu,
            "run_macos": self.run_macos,
            "reason": "; ".join(self.reasons),
        }


_DOC_FILES = frozenset(
    {
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "LICENSE-APACHE",
        "LICENSE-MIT",
        "README.md",
    }
)
_DOC_PREFIXES = ("docs/",)
_CI_FILES = frozenset(
    {
        ".github/dependabot.yml",
        "scripts/check-pr-fast.sh",
        "scripts/check-public-boundary-inventory.py",
        "scripts/test-public-boundary-inventory.py",
    }
)
_CI_PREFIXES = (".github/workflows/", "scripts/ci/")
_MACOS_CONTROL_FILES = frozenset(
    {
        ".github/workflows/ci-pr-workspace-tests.yml",
        "scripts/ci/change_policy.py",
        "scripts/ci/run_profile.py",
    }
)
_GPU_CONTROL_FILES = frozenset(
    {
        ".github/workflows/CI_gpu.yml",
        ".github/workflows/runpod-gpu-test.yml",
        ".github/workflows/ci-cache-publish.yml",
        "scripts/ci/change_policy.py",
        "scripts/ci/runpod_client.py",
        "scripts/ci/runpod_config.json",
        "scripts/ci/runpod_contract.py",
        "scripts/ci/recover_runpod_pr.py",
        "scripts/ci/find_archive_artifact.py",
        "scripts/ci/install_cuda_toolkit_hosted.sh",
        "scripts/ci/install_cuda_runtime_tree.sh",
        "scripts/ci/install_cutensor.sh",
        "scripts/ci/runpod_pricing.py",
        "scripts/ci/runpod_provision.py",
        "scripts/ci/cuda_smoke_test.py",
    }
)


def _is_docs_path(path: str) -> bool:
    return path in _DOC_FILES or path.startswith(_DOC_PREFIXES)


def _is_ci_path(path: str) -> bool:
    return path in _CI_FILES or path.startswith(_CI_PREFIXES)


def _is_gpu_control_plane_path(path: str) -> bool:
    return path in _GPU_CONTROL_FILES


def _is_macos_control_plane_path(path: str) -> bool:
    return path in _MACOS_CONTROL_FILES


def _full_policy(reason: str, *, run_gpu: bool = True) -> ChangePolicy:
    return ChangePolicy(
        change_class=ChangeClass.CODE,
        run_rust=True,
        run_blas=True,
        run_extensions=True,
        run_docs=True,
        run_ci_config=True,
        run_gpu=run_gpu,
        run_macos=True,
        reasons=(reason,),
    )


def classify_paths(
    paths: Sequence[str], event: str = "pull_request"
) -> ChangePolicy:
    """Classify paths, defaulting empty and unknown diffs to full validation."""

    normalized = tuple(
        sorted(
            {
                path.strip().removeprefix("./")
                for path in paths
                if path.strip()
            }
        )
    )
    if event == "push":
        return _full_policy("push-to-main override", run_gpu=False)
    if not normalized:
        return _full_policy("empty diff fallback")

    docs = tuple(path for path in normalized if _is_docs_path(path))
    ci = tuple(path for path in normalized if _is_ci_path(path))
    known = frozenset((*docs, *ci))
    unknown = tuple(path for path in normalized if path not in known)
    if unknown:
        return _full_policy("code or unknown paths: " + ", ".join(unknown))

    reasons = tuple(
        reason
        for reason in (
            "docs: " + ", ".join(docs) if docs else "",
            "ci: " + ", ".join(ci) if ci else "",
        )
        if reason
    )
    has_ci = bool(ci)
    return ChangePolicy(
        change_class=ChangeClass.CI_ONLY if has_ci else ChangeClass.DOCS_ONLY,
        run_rust=False,
        run_blas=False,
        run_extensions=False,
        run_docs=bool(docs),
        run_ci_config=has_ci,
        run_gpu=any(_is_gpu_control_plane_path(path) for path in ci),
        run_macos=any(_is_macos_control_plane_path(path) for path in ci),
        reasons=reasons,
    )


def _changed_paths(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base, head, "--"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _write_github_output(policy: ChangePolicy, output_path: str) -> None:
    output = policy.as_output()
    with Path(output_path).open("a", encoding="utf-8") as stream:
        for key, value in output.items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            if "\n" in rendered or "\r" in rendered:
                raise ValueError(f"multiline GitHub output is not supported: {key}")
            stream.write(f"{key}={rendered}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event", choices=("pull_request", "push"), default="pull_request"
    )
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--base")
    parser.add_argument("--head")
    args = parser.parse_args()
    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be provided together")
    if args.path and args.base:
        parser.error("use either --path or --base/--head")
    return args


def main() -> int:
    args = _parse_args()
    paths = _changed_paths(args.base, args.head) if args.base else args.path
    policy = classify_paths(paths, event=args.event)
    print(json.dumps(policy.as_output(), sort_keys=True))
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        _write_github_output(policy, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
