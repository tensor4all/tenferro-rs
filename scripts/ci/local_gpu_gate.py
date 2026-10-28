#!/usr/bin/env python3
"""Satisfy the required `CI GPU gate` from a maintainer's local GPU run.

A pull request labelled `gpu-validated-locally` gets no RunPod pod: the RunPod
workflow skips every job for it and does not publish `CI GPU gate`. The gate is
published instead from a run of the GPU lane on a maintainer's own GPU, bound to
the exact head SHA that was tested (REPOSITORY_RULES.md, CI Cost Discipline).

    python3 scripts/ci/local_gpu_gate.py run                 # on the GPU host, at the PR head
    python3 scripts/ci/local_gpu_gate.py publish --pr 1234   # after the head is pushed

`run` builds the same nextest archives the hosted CUDA archive job builds and runs
the same partitions (`gpu_test_partition.py`, host and device lanes, CUDA and
PJRT) plus the CUDA tutorial, then writes an evidence record keyed by the commit.
The PJRT device lane needs the JAX CUDA plugin and the cuDNN/NVCC wheels the pod
installs: point `--pjrt-root` (or `TENFERRO_LOCAL_PJRT_ROOT`) at a
`site-packages` that holds `jax_plugins/xla_cuda12`, `nvidia/cudnn/lib` and
`nvidia/cuda_nvcc`. `publish` posts that record as a PR
comment and dispatches `local-gpu-gate.yml` on trusted `main`, whose `relay`
step re-verifies everything and publishes the check. The publication has to come
from an Actions run: branch protection pins `CI GPU gate` to the GitHub Actions
app, and an account token can create neither a check run nor a status that
satisfies an app-pinned check.

A push moves the head, the published check stays on the old SHA, and the new head
needs a new local run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

LABEL = "gpu-validated-locally"
MARKER_PREFIX = "local-gpu-gate:v1"
WORKFLOW = "local-gpu-gate.yml"
CHECK_NAME = "CI GPU gate"
ALLOWED_ROLES = ("admin", "maintain")
SUMMARY_LIMIT = 60000

ROOT = Path(__file__).resolve().parents[2]
CUDA_CRATES = ("tenferro-gpu", "tenferro-ad", "tenferro-linalg", "tenferro-fft")

Runner = Callable[..., subprocess.CompletedProcess[str]]


def marker_line(sha: str) -> str:
    """The first line of a passing evidence comment for `sha`."""

    return f"<!-- {MARKER_PREFIX}:{sha}:passed -->"


def gate_external_id(sha: str) -> str:
    """The check run's `external_id`; never a `runpod-gpu-gate` (reusable) marker."""

    return f"{MARKER_PREFIX}:{sha}"


# --- relay (runs in local-gpu-gate.yml on trusted main) ----------------------


def relay_errors(
    *,
    ref: str,
    repository: str,
    actor: str,
    actor_role: str,
    head_sha: str,
    pr: dict,
    labels: Sequence[str],
    comment: dict,
    pr_number: int,
) -> list[str]:
    """Every reason the inputs do not authorize a passing `CI GPU gate`."""

    errors = []
    if ref != "refs/heads/main":
        errors.append(f"the relay must run from trusted main, not {ref}")
    if actor_role not in ALLOWED_ROLES:
        errors.append(f"{actor} has role '{actor_role}' (requires admin or maintain)")
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        errors.append(f"head_sha is not a full commit SHA: {head_sha!r}")
    if pr.get("state") != "open":
        errors.append(f"PR #{pr_number} is {pr.get('state')}")
    head = pr.get("head") or {}
    if (head.get("repo") or {}).get("full_name") != repository:
        errors.append(f"PR #{pr_number} does not come from {repository}")
    if head.get("sha") != head_sha:
        errors.append(
            f"PR #{pr_number} head is {head.get('sha')}, not the tested {head_sha}"
        )
    if LABEL not in labels:
        errors.append(f"PR #{pr_number} is not labelled {LABEL}")
    issue_url = comment.get("issue_url", "")
    if not issue_url.endswith(f"/issues/{pr_number}"):
        errors.append(f"the evidence comment does not belong to PR #{pr_number}")
    if (comment.get("user") or {}).get("login") != actor:
        errors.append("the evidence comment was not written by the dispatching account")
    body = comment.get("body") or ""
    if body.splitlines()[:1] != [marker_line(head_sha)]:
        errors.append(f"the evidence comment does not record a pass on {head_sha}")
    return errors


def check_run_payload(*, head_sha: str, comment: dict, run_url: str) -> dict:
    """The `CI GPU gate` check run published for a verified local run."""

    body = comment.get("body") or ""
    summary = f"Evidence: {comment.get('html_url', '')}\n\n{body}"
    return {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": "success",
        # Run-shaped, so `CI_gpu.yml`'s poller treats it as a verdict rather than
        # one of its own jobs.
        "details_url": run_url,
        "external_id": gate_external_id(head_sha),
        "output": {
            "title": "CI GPU gate passed (local GPU run)",
            "summary": summary[:SUMMARY_LIMIT],
        },
    }


def gh_api(path: str, *, runner: Runner = subprocess.run, method: str = "GET",
           payload: dict | None = None) -> object:
    command = ["gh", "api", "--method", method, path]
    if payload is not None:
        command += ["--input", "-"]
    done = runner(
        command,
        input=json.dumps(payload) if payload is not None else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(done.stdout) if done.stdout.strip() else None


def role_of(repository: str, login: str, runner: Runner = subprocess.run) -> str:
    data = gh_api(f"repos/{repository}/collaborators/{login}/permission", runner=runner)
    return str((data or {}).get("role_name", "")) if isinstance(data, dict) else ""


def relay(env: dict[str, str], runner: Runner = subprocess.run) -> int:
    repository = env["GITHUB_REPOSITORY"]
    actor = env["GITHUB_ACTOR"]
    pr_number = int(env["PR_NUMBER"])
    head_sha = env["HEAD_SHA"].strip()
    comment_id = int(env["COMMENT_ID"])

    def inputs() -> dict:
        pr = gh_api(f"repos/{repository}/pulls/{pr_number}", runner=runner)
        labels = gh_api(f"repos/{repository}/issues/{pr_number}/labels", runner=runner)
        comment = gh_api(
            f"repos/{repository}/issues/comments/{comment_id}", runner=runner
        )
        return {
            "pr": pr,
            "labels": [label["name"] for label in labels],
            "comment": comment,
        }

    common = {
        "ref": env["GITHUB_REF"],
        "repository": repository,
        "actor": actor,
        "actor_role": role_of(repository, actor, runner),
        "head_sha": head_sha,
        "pr_number": pr_number,
    }
    state = inputs()
    errors = relay_errors(**common, **state)
    if not errors:
        # Re-read right before publishing: a push or a label change between the
        # first read and the POST must not publish onto a head nobody tested.
        state = inputs()
        errors = relay_errors(**common, **state)
    if errors:
        for error in errors:
            print(f"::error::{error}")
        return 1
    payload = check_run_payload(
        head_sha=head_sha, comment=state["comment"], run_url=env["RUN_URL"]
    )
    gh_api(f"repos/{repository}/check-runs", runner=runner, method="POST",
           payload=payload)
    print(f"Published {CHECK_NAME} for {head_sha} from {state['comment']['html_url']}.")
    return 0


# --- run (on the GPU host) ----------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def evidence_dir() -> Path:
    return ROOT / "target" / "local-gpu-gate"


def nextest_summaries(text: str) -> list[str]:
    """The nextest `Summary [...]` lines of one run's output."""

    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return [line.strip() for line in plain.splitlines()
            if re.match(r"^\s*Summary \[", line)]


def stream(command: Sequence[str], env: dict[str, str], log: Path) -> tuple[int, str]:
    """Run `command`, echoing and logging its output; return the code and output."""

    print(f"$ {shlex.join(command)}", flush=True)
    captured = []
    with log.open("a") as sink, subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    ) as process:
        sink.write(f"$ {shlex.join(command)}\n")
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sink.write(line)
            captured.append(line)
        code = process.wait()
    return code, "".join(captured)


def gpu_facts() -> dict[str, str]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    header = subprocess.run(["nvidia-smi"], check=True, capture_output=True,
                            text=True).stdout
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    busy = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    toolkit = ""
    nvcc = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")) / "bin" / "nvcc"
    if nvcc.exists():
        out = subprocess.run([str(nvcc), "--version"], capture_output=True,
                             text=True).stdout
        found = re.search(r"release ([0-9.]+)", out)
        toolkit = found.group(1) if found else ""
    return {
        "gpus": "; ".join(line.strip() for line in query),
        "driver_cuda": cuda.group(1) if cuda else "unknown",
        "toolkit": toolkit or "unknown",
        "busy": busy,
    }


def pjrt_env(root: Path, env: dict[str, str]) -> dict[str, str] | None:
    """The PJRT device lane's environment, as runpod-gpu-execute.yml sets it up."""

    plugin = root / "jax_plugins" / "xla_cuda12" / "xla_cuda_plugin.so"
    cudnn = root / "nvidia" / "cudnn" / "lib"
    nvcc = root / "nvidia" / "cuda_nvcc"
    if not (plugin.is_file() and cudnn.is_dir()
            and (nvcc / "nvvm" / "libdevice" / "libdevice.10.bc").is_file()):
        return None
    return dict(
        env,
        NEXTEST_TEST_THREADS="1",
        TENFERRO_PJRT_PLUGIN=str(plugin),
        XLA_FLAGS=f"--xla_gpu_cuda_data_dir={nvcc} {env.get('XLA_FLAGS', '')}".strip(),
        LD_LIBRARY_PATH=f"{cudnn}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":"),
    )


def target_directory() -> Path:
    """Cargo's target directory for this workspace (honours CARGO_TARGET_DIR)."""

    metadata = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return Path(json.loads(metadata)["target_directory"])


def lane_steps(
    env: dict[str, str],
    device_pjrt: dict[str, str],
    jobs: int,
    target: Path,
    out: Path,
) -> list[tuple[str, list[str], dict[str, str]]]:
    """The GPU lane, step by step, as the hosted archive job and the pod run it."""

    cuda_archive = out / "cuda-tests.tar.zst"
    pjrt_archive = out / "pjrt-tests.tar.zst"
    host_env = dict(env, CUDA_VISIBLE_DEVICES="", RAYON_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                    NEXTEST_TEST_THREADS=str(jobs))
    partition = ["python3", "scripts/ci/gpu_test_partition.py"]
    packages = [arg for name in CUDA_CRATES for arg in ("-p", name)]
    return [
        ("CUDA archive", ["cargo", "nextest", "archive",
                          "--cargo-profile", "ci", "--archive-file", str(cuda_archive),
                          *packages, "--features", "cuda"], env),
        ("PJRT archive", ["cargo", "nextest", "archive",
                          "--cargo-profile", "ci", "--archive-file", str(pjrt_archive),
                          "-p", "tenferro-xla", "--features", "pjrt"], env),
        ("CUDA tutorial build", ["cargo", "build", "-j", str(jobs), "--profile", "ci",
                                 "--manifest-path", "docs/tutorial-code/Cargo.toml",
                                 "--no-default-features", "--features", "cuda,native",
                                 "--bin", "cuda_tutorial"], env),
        ("CUDA host lane", [*partition, "--kind", "cuda", "--lane", "host",
                            "--archive-file", str(cuda_archive)], host_env),
        ("PJRT host lane", [*partition, "--kind", "pjrt", "--lane", "host",
                            "--archive-file", str(pjrt_archive)], host_env),
        ("CUDA device lane", [*partition, "--kind", "cuda", "--lane", "gpu",
                              "--archive-file", str(cuda_archive)],
         dict(env, NEXTEST_TEST_THREADS="1")),
        ("CUDA tutorial", [str(target / "ci" / "cuda_tutorial")],
         dict(env, TENFERRO_REQUIRE_CUDA="1")),
        ("PJRT device lane", [*partition, "--kind", "pjrt", "--lane", "gpu",
                              "--archive-file", str(pjrt_archive)], device_pjrt),
    ]


def run(args: argparse.Namespace) -> int:
    if dirty := git("status", "--porcelain"):
        print(f"The worktree is not clean; commit first:\n{dirty}", file=sys.stderr)
        return 2
    sha = git("rev-parse", "HEAD")
    pjrt_root = args.pjrt_root or os.environ.get("TENFERRO_LOCAL_PJRT_ROOT", "")
    device_pjrt = pjrt_env(Path(pjrt_root), dict(os.environ)) if pjrt_root else None
    if device_pjrt is None:
        print("The PJRT device lane is part of the GPU lane: pass --pjrt-root (or set "
              "TENFERRO_LOCAL_PJRT_ROOT) to a site-packages with jax_plugins/xla_cuda12, "
              "nvidia/cudnn/lib and nvidia/cuda_nvcc.", file=sys.stderr)
        return 2
    facts = gpu_facts()
    if facts["busy"] and not args.allow_busy_gpu:
        print(f"The GPU is in use by other processes:\n{facts['busy']}\n"
              "Wait, or pass --allow-busy-gpu.", file=sys.stderr)
        return 2
    out = evidence_dir()
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{sha}.log"
    log.write_text("")

    env = dict(os.environ)
    # The hosted archive job builds with these bindings (runpod-gpu-test.yml).
    env.setdefault("CUDARC_CUDA_VERSION", "12080")
    env.setdefault(
        "TENFERRO_CUDA_ARTIFACT_CACHE",
        str(Path.home() / ".cache" / "tenferro" / "t4a-cuda-artifacts"),
    )
    # `cargo nextest archive` takes no `-j`; the build reads CARGO_BUILD_JOBS.
    env["CARGO_BUILD_JOBS"] = str(args.jobs)
    # Uncoloured, so the nextest summaries can be read back from the output.
    env["CARGO_TERM_COLOR"] = "never"
    device_pjrt = pjrt_env(Path(pjrt_root), env)
    assert device_pjrt is not None
    steps = lane_steps(env, device_pjrt, args.jobs, target_directory(), out)

    results = []
    passed = True
    for name, command, step_env in steps:
        code, output = stream(command, step_env, log)
        summary = nextest_summaries(output)
        results.append((name, command, code, summary))
        if code != 0:
            passed = False
            break
    # The build and tests read the worktree for a long time; evidence is valid
    # only if what they read is still exactly the commit it names.
    if passed and (git("rev-parse", "HEAD") != sha or git("status", "--porcelain")):
        print("HEAD or the worktree changed during the run; the evidence is void. "
              "Leave the worktree untouched while `run` executes.", file=sys.stderr)
        return 2

    verdict = "passed" if passed else "failed"
    lines = [
        marker_line(sha) if passed else f"<!-- {MARKER_PREFIX}:{sha}:failed -->",
        f"Local GPU gate: **{verdict}** on `{sha}`",
        "",
        f"- Host: `{socket.gethostname()}`",
        f"- GPU: {facts['gpus']} (driver CUDA {facts['driver_cuda']}, toolkit {facts['toolkit']})",
        f"- `CUDARC_CUDA_VERSION={env['CUDARC_CUDA_VERSION']}`, "
        f"`TENFERRO_CUDA_ARTIFACT_CACHE={env['TENFERRO_CUDA_ARTIFACT_CACHE']}`",
        "",
        "| step | result | nextest summary |",
        "|---|---|---|",
    ]
    for name, _, code, summary in results:
        lines.append(f"| {name} | {'ok' if code == 0 else f'exit {code}'} | "
                     f"{'<br>'.join(summary)} |")
    ran = {name for name, *_ in results}
    for name, *_ in steps:
        if name not in ran:
            lines.append(f"| {name} | NOT RUN (an earlier step failed) | |")
    lines += ["", "Commands:", "", "```text"]
    lines += [shlex.join(command) for _, command, *_ in results]
    lines += ["```"]
    record = {
        "sha": sha,
        "passed": passed,
        "body": "\n".join(lines) + "\n",
    }
    (out / f"{sha}.json").write_text(json.dumps(record, indent=2))
    print(record["body"])
    print(f"Evidence: {out / f'{sha}.json'} (log {log})")
    return 0 if passed else 1


# --- publish (after the head is pushed) ----------------------------------------


def publish_errors(*, record: dict, pr: dict, labels: Sequence[str], role: str,
                   login: str, repository: str) -> list[str]:
    """Every reason `publish` must not post this evidence to this PR."""

    errors = []
    if role not in ALLOWED_ROLES:
        errors.append(f"{login} has role '{role}' (requires admin or maintain)")
    if not record.get("passed"):
        errors.append("the recorded local run did not pass")
    if pr.get("state") != "open":
        errors.append(f"the PR is {pr.get('state')}")
    head = pr.get("head") or {}
    if (head.get("repo") or {}).get("full_name") != repository:
        errors.append("fork PRs keep the hosted gate")
    if head.get("sha") != record.get("sha"):
        errors.append(f"the PR head is {head.get('sha')}, the evidence is for "
                      f"{record.get('sha')}; rerun `run` at the PR head")
    if LABEL not in labels:
        errors.append(f"the PR is not labelled {LABEL}")
    return errors


def publish(args: argparse.Namespace, runner: Runner = subprocess.run) -> int:
    repository = args.repo
    sha = args.sha or git("rev-parse", "HEAD")
    record = json.loads((evidence_dir() / f"{sha}.json").read_text())
    login = str(gh_api("user", runner=runner)["login"])  # type: ignore[index]
    pr = gh_api(f"repos/{repository}/pulls/{args.pr}", runner=runner)
    labels = [label["name"] for label in
              gh_api(f"repos/{repository}/issues/{args.pr}/labels", runner=runner)]
    errors = publish_errors(record=record, pr=pr, labels=labels,
                            role=role_of(repository, login, runner), login=login,
                            repository=repository)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    comment = gh_api(f"repos/{repository}/issues/{args.pr}/comments", runner=runner,
                     method="POST", payload={"body": record["body"]})
    comment_id = comment["id"]  # type: ignore[index]
    runner(["gh", "workflow", "run", WORKFLOW, "--repo", repository, "--ref", "main",
            "-f", f"pr_number={args.pr}", "-f", f"head_sha={sha}",
            "-f", f"comment_id={comment_id}"], check=True, text=True)
    print(f"Posted {comment['html_url']} and dispatched {WORKFLOW};"  # type: ignore[index]
          f" `{CHECK_NAME}` appears on {sha} when it finishes.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run the GPU lane at HEAD and record evidence")
    run_parser.add_argument("--jobs", type=int, default=4, help="cargo build jobs")
    run_parser.add_argument("--allow-busy-gpu", action="store_true")
    run_parser.add_argument("--pjrt-root", help="site-packages holding the PJRT plugin")
    publish_parser = sub.add_parser("publish", help="post the evidence and relay the gate")
    publish_parser.add_argument("--pr", type=int, required=True)
    publish_parser.add_argument("--sha", help="evidence commit (default: HEAD)")
    publish_parser.add_argument("--repo", default="tensor4all/tenferro-rs")
    sub.add_parser("relay", help="verify and publish (local-gpu-gate.yml only)")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "publish":
        return publish(args)
    return relay(dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
