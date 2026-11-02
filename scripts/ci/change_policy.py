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
        "AGENTS.md",
        "CHANGELOG.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "CONTRIBUTORS.md",
        "GOVERNANCE.md",
        "LICENSE-APACHE",
        "LICENSE-MIT",
        "PERFORMANCE_TIPS.md",
        "README.md",
        "REPOSITORY_RULES.md",
    }
)
# `ai/` holds the contribution workflows and agent prompts. They are prose about
# how work is done and cannot change what any lane executes, so editing one must
# not reserve the paid GPU lane the way an unknown path does.
_DOC_PREFIXES = ("docs/", "ai/")
_CI_FILES = frozenset(
    {
        ".github/dependabot.yml",
        "scripts/check-pr-fast.sh",
        "scripts/check-contraction-kernels.py",
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
# Paid-lane control plane, split by what a diff can actually change.
#
# `_GPU_DEVICE_CONTROL_FILES` changes what the pod executes or whether it can
# execute at all: the archive and its payload, the test partitions, the pod
# request/provision/staging path, and the cache the pod consumes. A diff
# touching one of them is proven on a pod.
_GPU_DEVICE_CONTROL_FILES = frozenset(
    {
        ".github/workflows/runpod-gpu-test.yml",
        ".github/workflows/runpod-gpu-execute.yml",
        ".github/workflows/runpod-gpu-runtime.yml",
        ".github/workflows/ci-cache-publish.yml",
        "scripts/ci/change_policy.py",
        "scripts/ci/runpod_client.py",
        "scripts/ci/runpod_http.py",
        "scripts/ci/runpod_config.json",
        "scripts/ci/runpod_contract.py",
        "scripts/ci/recover_runpod_pr.py",
        "scripts/ci/find_archive_artifact.py",
        "scripts/ci/gpu_test_partition.py",
        "scripts/ci/gpu_nextest.toml",
        "scripts/ci/cuda_test_partition.tsv",
        "scripts/ci/pjrt_test_partition.tsv",
        "scripts/ci/install_cuda_toolkit_hosted.sh",
        "scripts/ci/install_cuda_runtime_tree.sh",
        "scripts/ci/prepare_gpu_execution_payload.sh",
        "scripts/ci/check_cuda_headers.py",
        "scripts/ci/install_cutensor.sh",
        "scripts/ci/runpod_provision.py",
        "scripts/ci/runpod_setup_watchdog.py",
        "scripts/ci/runpod_lifecycle.py",
        "scripts/ci/cuda_smoke_test.py",
    }
)

# The paid lane's host-side half: how the gate is waited on, reused, priced, and
# reclaimed, and the local GPU gate that replaces the pod for a labelled PR.
# None of it changes what the pod executes, and running the device suite cannot
# observe any of it - `test_gpu_gate_poller.py`, `test_gpu_gate_reuse.py` and
# `test_local_gpu_gate.py` are the checks that can. A CI-only diff that touches
# only these therefore skips the paid path, and the gate publishes "not
# required" instead of provisioning a pod (REPOSITORY_RULES.md, CI Cost
# Discipline). `.github/workflows/CI_gpu.yml` is the required-gate contract plus
# the fork lane, which runs on GitHub-hosted and `ubuntu-gpu` runners, never on
# a RunPod pod; #2163 spent a pod on a wait-loop fix before this split existed.
_GPU_HOSTED_ONLY_FILES = frozenset(
    {
        ".github/workflows/CI_gpu.yml",
        ".github/workflows/local-gpu-gate.yml",
        ".github/workflows/runpod-reap.yml",
        "scripts/ci/gpu_gate_reuse.py",
        "scripts/ci/local_gpu_gate.py",
        "scripts/ci/runpod_pricing.py",
        "scripts/ci/runpod_cost.py",
        "scripts/ci/runpod_reap.py",
    }
)


# Markdown is prose wherever it lives: a README under `docs/tutorial-code/` or a
# release note under `.github/releases/` cannot change what any lane executes,
# so editing one must not reserve the paid GPU lane (#2376).
_DOC_MARKDOWN_PREFIXES = ("docs/", ".github/releases/")


def _is_docs_path(path: str) -> bool:
    if path.endswith(".md") and path.startswith(_DOC_MARKDOWN_PREFIXES):
        return True
    # This tree is a workspace member and supplies the CUDA tutorial artifact.
    # Treat its sources, manifest, and unknown additions as code, not prose.
    if path.startswith("docs/tutorial-code/"):
        return False
    return path in _DOC_FILES or path.startswith(_DOC_PREFIXES)


def _is_ci_path(path: str) -> bool:
    return path in _CI_FILES or path.startswith(_CI_PREFIXES)


def _is_gpu_device_control_plane_path(path: str) -> bool:
    return path in _GPU_DEVICE_CONTROL_FILES


def _is_macos_control_plane_path(path: str) -> bool:
    return path in _MACOS_CONTROL_FILES


_RUST_SUFFIX = ".rs"
# A changed Rust line may only belong to one of these shapes to count as
# documentation: blank, `///`/`//!` comment, `#[doc ...]` attribute, or a
# `/** ... */` block-comment line. Everything else keeps the full policy.
_DOC_LINE_PREFIXES = ("///", "//!", "#[doc", "/**", "*/")
_TEST_DIRECTORIES = frozenset({"tests", "benches", "examples"})


def _is_rust_comment_only_line(line: str) -> bool:
    """Return True when one changed Rust line only edits documentation."""

    text = line.strip()
    if not text:
        return True
    if text.startswith(_DOC_LINE_PREFIXES):
        return True
    # Block-comment body: accept a bare '*' or '* text', never '*ptr = 1;'.
    return text == "*" or text.startswith(("* ", "*\t"))


def _is_rust_test_path(path: str) -> bool:
    """Test, bench and example targets always keep the full policy."""

    parts = Path(path).parts
    name = parts[-1]
    return (
        any(part in _TEST_DIRECTORIES for part in parts[:-1])
        or name in {"test.rs", "tests.rs"}
        or name.startswith("test_")
        or name.endswith(("_test.rs", "_tests.rs"))
    )


def _split_diff_blocks(diff_text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            blocks.append([])
        if blocks:
            blocks[-1].append(line)
    return blocks


def _diff_block_path(block: Sequence[str]) -> str | None:
    for line in block:
        if line.startswith("+++ b/"):
            return line[len("+++ b/") :]
    for line in block:
        if line.startswith("--- a/"):
            return line[len("--- a/") :]
    return None


def _diff_block_is_docs_only(block: Sequence[str]) -> bool:
    in_hunk = False
    saw_change = False
    for line in block:
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("\\"):
            # "\ No newline at end of file" carries no source text.
            continue
        if line[:1] in {"+", "-", " "}:
            saw_change = True
            if not _is_rust_comment_only_line(line[1:]):
                return False
            continue
        return False
    # An empty diff (pure rename, identical content) proves nothing.
    return saw_change


def _parse_docs_only_diff(
    diff_text: str, candidates: Sequence[str]
) -> frozenset[str]:
    allowed = frozenset(candidates)
    verdicts: dict[str, bool] = {}
    for block in _split_diff_blocks(diff_text):
        path = _diff_block_path(block)
        if path is None or path not in allowed:
            # Unexpected shape (quoted or unparsable path): keep full policy.
            return frozenset()
        verdicts[path] = verdicts.get(path, True) and _diff_block_is_docs_only(
            block
        )
    return frozenset(path for path, docs_only in verdicts.items() if docs_only)


def _docs_only_rust_paths(
    paths: Sequence[str],
    base: str,
    head: str,
    *,
    cwd: str | None = None,
) -> frozenset[str]:
    """Return the Rust paths whose entire diff only edits documentation.

    Fail closed: any non-Rust path, test target, git failure or unexpected diff
    shape yields an empty set, so the caller keeps the full policy.
    """

    candidates = tuple(
        path
        for path in paths
        if path.endswith(_RUST_SUFFIX) and not _is_rust_test_path(path)
    )
    if not candidates or len(candidates) != len(paths):
        return frozenset()
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "-U0",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                f"{base}...{head}",
                "--",
                *candidates,
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return _parse_docs_only_diff(result.stdout, candidates)


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
    paths: Sequence[str],
    event: str = "pull_request",
    *,
    base: str | None = None,
    head: str | None = None,
    cwd: str | None = None,
) -> ChangePolicy:
    """Classify paths, defaulting empty and unknown diffs to full validation.

    `base`/`head` are optional. When both are given, an otherwise-unknown diff
    that only edits Rust documentation comments keeps every lane except the
    paid GPU pod. Without them the diff content cannot be inspected and the
    full policy stands (fail closed).
    """

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
        comment_only = (
            _docs_only_rust_paths(unknown, base, head, cwd=cwd)
            if base and head
            else frozenset()
        )
        if comment_only and len(comment_only) == len(unknown):
            # Only the paid device lane depends on what the code says. A
            # GPU control-plane file in the same diff still reserves the pod.
            return ChangePolicy(
                change_class=ChangeClass.CODE,
                run_rust=True,
                run_blas=True,
                run_extensions=True,
                run_docs=True,
                run_ci_config=True,
                run_gpu=any(
                    _is_gpu_device_control_plane_path(path) for path in ci
                ),
                run_macos=True,
                reasons=tuple(
                    reason
                    for reason in (
                        "docs-only Rust comments: "
                        + ", ".join(sorted(comment_only)),
                        "docs: " + ", ".join(docs) if docs else "",
                        "ci: " + ", ".join(ci) if ci else "",
                    )
                    if reason
                ),
            )
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
        run_gpu=any(_is_gpu_device_control_plane_path(path) for path in ci),
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
    parser.add_argument(
        "--path", action="append", default=[], help="changed path (repeatable)"
    )
    parser.add_argument("--base", help="base revision of the diff")
    parser.add_argument(
        "--head",
        help="head revision; with --path it only enables content inspection",
    )
    args = parser.parse_args()
    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be provided together")
    return args


def main() -> int:
    args = _parse_args()
    if args.path:
        paths = list(args.path)
    elif args.base:
        paths = _changed_paths(args.base, args.head)
    else:
        paths = []
    policy = classify_paths(
        paths, event=args.event, base=args.base, head=args.head
    )
    print(json.dumps(policy.as_output(), sort_keys=True))
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        _write_github_output(policy, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
