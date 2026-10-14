"""The CI GPU gate's wait loop must survive a transient checks-API failure.

`CI_gpu.yml`'s "Require fork CUDA GPU tests or same-repo RunPod gate" step waits
up to 90 minutes for the RunPod workflow to publish the `CI GPU gate` check, and
that check is required by branch protection. Two defects made it conclude a gate
that had no verdict:

* it ran `gh api ... --jq .` straight into `jq` under `set -euo pipefail`, so one
  empty body or 5xx from the checks API concluded the gate red (2026-10-14,
  PR #2162);
* it treated *any* `CI GPU gate` check-run outside the current run as the
  verdict, and every run of its own workflow publishes one too. A re-run read a
  cancelled job from the previous attempt and failed in three seconds (#2163).

The test extracts the step's script, runs it against a stub `gh`, and covers the
retry, the identification of the RunPod publication, a completed failure, and the
policy short-circuit that keeps a host-side-only CI diff out of the paid lane.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "CI_gpu.yml"
STEP_NAME = "Require fork CUDA GPU tests or same-repo RunPod gate"

RUN_URL = "https://example/actions/runs/4242"
JOB_URL = "https://example/actions/runs/999/job/77"


def check_run(conclusion: str, *, details_url: str, status: str = "completed") -> dict:
    return {
        "status": status,
        "conclusion": conclusion if status == "completed" else None,
        "details_url": details_url,
        "started_at": "2026-01-01T00:00:00Z",
    }


def extract_step_script() -> str:
    """Return the `run:` body of the gate step, dedented."""
    lines = WORKFLOW.read_text().splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == f"- name: {STEP_NAME}"
    )
    run_at = next(
        index for index in range(start, len(lines)) if lines[index].strip() == "run: |"
    )
    indent = len(lines[run_at]) - len(lines[run_at].lstrip())
    body: list[str] = []
    for line in lines[run_at + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line[indent + 2 :] if len(line) > indent + 2 else "")
    return textwrap.dedent("\n".join(body)) + "\n"


class GatePollerTests(unittest.TestCase):
    def setUp(self) -> None:
        real_jq = shutil.which("jq")
        if real_jq is None:
            self.skipTest("jq is not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.state = root / "state"
        self.state.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()

        # The step asks the checks API for the gate and the pulls API for the
        # changed files. The stub answers both; `fail_until` counts every call,
        # so a transient failure is expressed by leaving it above the number of
        # requests the tick makes before it reaches the checks API.
        (self.bin / "gh").write_text(
            textwrap.dedent(
                """\
                #!/bin/bash
                n=$(cat "${GATE_STATE}/calls"); n=$((n + 1)); echo "${n}" > "${GATE_STATE}/calls"
                if [ "${n}" -le "$(cat "${GATE_STATE}/fail_until")" ]; then
                  echo "gh: HTTP 502" >&2
                  exit 1
                fi
                if [[ "$*" == *"/files"* ]]; then
                  cat "${GATE_STATE}/files.txt"
                else
                  cat "${GATE_STATE}/checks.json"
                fi
                """
            )
        )
        (self.bin / "jq").write_text(
            textwrap.dedent(
                f"""\
                #!/bin/bash
                last="${{@: -1}}"
                if [ -f "${{last}}" ]; then
                  if [ ! -s "${{last}}" ]; then exit 5; fi
                  if [[ "${{last}}" == *checks.json ]] && ! grep -q '"check_runs"' "${{last}}"; then
                    exit 5
                  fi
                fi
                exec "{real_jq}" "$@"
                """
            )
        )
        (self.bin / "sleep").write_text("#!/bin/bash\nexit 0\n")
        for name in ("gh", "jq", "sleep"):
            (self.bin / name).chmod(0o755)

        self.script = root / "gate_step.sh"
        self.script.write_text(extract_step_script())

    def run_step(
        self,
        *,
        check_runs: list[dict] | None = None,
        files: list[str] | None = None,
        fail_until: int = 0,
        timeout: float = 30,
    ) -> tuple[int, str]:
        (self.state / "fail_until").write_text(str(fail_until))
        (self.state / "checks.json").write_text(
            json.dumps({"check_runs": check_runs if check_runs is not None else []})
        )
        (self.state / "files.txt").write_text("\n".join(files or []) + "\n")
        (self.state / "calls").write_text("0")
        env = dict(os.environ)
        env.update(
            PATH=f"{self.bin}{os.pathsep}{env['PATH']}",
            GATE_STATE=str(self.state),
            GITHUB_REPOSITORY="tensor4all/tenferro-rs",
            HEAD_SHA="0" * 40,
            CURRENT_RUN_MARKER="/actions/runs/999",
            HEAD_REPO="tensor4all/tenferro-rs",
            PR_NUMBER="2163",
        )
        try:
            done = subprocess.run(
                ["bash", str(self.script)],
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as expired:
            captured = b"".join(
                part if isinstance(part, bytes) else part.encode()
                for part in (expired.stdout, expired.stderr)
                if part
            )
            return 124, captured.decode(errors="replace")
        return done.returncode, done.stdout + done.stderr

    def test_transient_checks_api_failures_are_retried(self) -> None:
        # Call 1 is the changed-file list, which is guarded rather than
        # retried; call 2 is the checks request, which retries into call 3.
        code, output = self.run_step(
            check_runs=[check_run("success", details_url=RUN_URL)],
            files=["crates/tenferro-gpu/src/lib.rs"],
            fail_until=2,
        )
        self.assertEqual(code, 0, output)
        self.assertIn("RunPod CI GPU gate passed", output)
        self.assertIn("retrying", output)

    def test_a_run_of_this_workflow_is_not_a_runpod_verdict(self) -> None:
        """An earlier attempt's own gate job must not be read as the verdict."""
        code, output = self.run_step(
            check_runs=[
                check_run("cancelled", details_url=JOB_URL),
                check_run("success", details_url=RUN_URL),
            ],
            files=["crates/tenferro-gpu/src/lib.rs"],
        )
        self.assertEqual(code, 0, output)
        self.assertIn("RunPod CI GPU gate passed", output)

    def test_only_a_stale_job_check_keeps_waiting(self) -> None:
        # With nothing but this workflow's own job in the list there is no
        # verdict yet, so the loop must keep waiting rather than fail.
        code, output = self.run_step(
            check_runs=[check_run("cancelled", details_url=JOB_URL)],
            files=["crates/tenferro-gpu/src/lib.rs"],
            timeout=8,
        )
        self.assertEqual(code, 124, output)
        self.assertNotIn("concluded", output)

    def test_a_completed_failure_is_still_reported(self) -> None:
        code, output = self.run_step(
            check_runs=[check_run("failure", details_url=RUN_URL)],
            files=["crates/tenferro-gpu/src/lib.rs"],
        )
        self.assertEqual(code, 1, output)
        self.assertIn("concluded failure", output)

    def test_host_side_only_ci_diff_skips_the_paid_lane(self) -> None:
        # `run_gpu=false` for this file list, so the required check concludes on
        # the host instead of waiting for a pod the change does not need.
        code, output = self.run_step(files=[".github/workflows/CI_gpu.yml"])
        self.assertEqual(code, 0, output)
        self.assertIn("GPU validation not required", output)
        self.assertNotIn("check-runs", (self.state / "calls").read_text() + output)

    def test_a_body_that_never_parses_does_not_fail_the_gate(self) -> None:
        (self.state / "files.txt").write_text("crates/tenferro-gpu/src/lib.rs\n")
        (self.state / "fail_until").write_text("0")
        (self.state / "calls").write_text("0")
        # An empty checks body is what `gh` returned on the incident this fixes.
        (self.state / "checks.json").write_text("")
        env = dict(os.environ)
        env.update(
            PATH=f"{self.bin}{os.pathsep}{env['PATH']}",
            GATE_STATE=str(self.state),
            GITHUB_REPOSITORY="tensor4all/tenferro-rs",
            HEAD_SHA="0" * 40,
            CURRENT_RUN_MARKER="/actions/runs/999",
            HEAD_REPO="tensor4all/tenferro-rs",
            PR_NUMBER="2163",
        )
        try:
            done = subprocess.run(
                ["bash", str(self.script)],
                capture_output=True,
                text=True,
                env=env,
                timeout=8,
            )
        except subprocess.TimeoutExpired as expired:
            output = (expired.stdout or b"").decode(errors="replace")
            self.assertIn("checks API is unavailable", output)
            return
        self.fail(f"the loop exited instead of waiting: {done.stdout}{done.stderr}")


if __name__ == "__main__":
    unittest.main()
