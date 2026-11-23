"""A fork PR's `CI GPU gate` passes only on a maintainer-approved RunPod verdict.

`scripts/ci/fork_gpu_gate.py` reads the verdict from the `fork-pr-verdict` job
of a manual dispatch of the main `runpod-gpu-test.yml`, and that job re-runs the
fork PR's gate when the gate already gave up. These tests pin which runs count,
that the job name binds the verdict to one PR head, and that a re-run never
turns a passed verdict red.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.ci.fork_gpu_gate import (
    RUNPOD_WORKFLOW_PATH,
    find_verdict,
    main,
    rerun_gate,
    verdict_job_name,
    wait_for_verdict,
)

ROOT = Path(__file__).resolve().parents[3]
REPO = "tensor4all/tenferro-rs"
PR = 2717
HEAD = "d" * 40
OTHER = "e" * 40


def runpod_run(run_id: int, **overrides: object) -> dict:
    run = {
        "id": run_id,
        "path": RUNPOD_WORKFLOW_PATH,
        "event": "workflow_dispatch",
        "head_branch": "main",
        "repository": {"full_name": REPO},
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }
    run.update(overrides)
    return run


def job(name: str, conclusion: str | None = "success") -> dict:
    return {"name": name, "conclusion": conclusion, "html_url": "https://job"}


class FakeGitHub:
    """Answers the two listings and records POSTs."""

    def __init__(
        self,
        runs: list[dict],
        jobs: dict[int, list[dict]] | None = None,
        *,
        gate_runs: list[dict] | None = None,
        post_status: int = 201,
        fail: bool = False,
    ) -> None:
        self.runs = runs
        self.jobs = jobs or {}
        self.gate_runs = gate_runs or []
        self.post_status = post_status
        self.fail = fail
        self.posts: list[str] = []

    def __call__(self, method: str, url: str) -> tuple[int, bytes]:
        if self.fail:
            return 502, b""
        if method == "POST":
            self.posts.append(url)
            return self.post_status, b""
        if "/workflows/runpod-gpu-test.yml/runs" in url:
            return 200, json.dumps({"workflow_runs": self.runs}).encode()
        if "/workflows/CI_gpu.yml/runs" in url:
            return 200, json.dumps({"workflow_runs": self.gate_runs}).encode()
        run_id = int(url.split("/actions/runs/", 1)[1].split("/", 1)[0])
        return 200, json.dumps({"jobs": self.jobs.get(run_id, [])}).encode()


class FindVerdictTests(unittest.TestCase):
    def test_a_passed_verdict_for_this_head_is_found(self) -> None:
        github = FakeGitHub([runpod_run(1)], {1: [job(verdict_job_name(PR, HEAD))]})
        self.assertEqual(find_verdict(github, REPO, PR, HEAD), "https://job")

    def test_another_head_or_pr_is_not_a_verdict(self) -> None:
        # A push after the approval leaves the old verdict on the old head.
        github = FakeGitHub(
            [runpod_run(1)],
            {1: [job(verdict_job_name(PR, OTHER)), job(verdict_job_name(PR + 1, HEAD))]},
        )
        self.assertIsNone(find_verdict(github, REPO, PR, HEAD))

    def test_only_a_successful_verdict_job_counts(self) -> None:
        for conclusion in ("failure", "skipped", None):
            with self.subTest(conclusion=conclusion):
                github = FakeGitHub(
                    [runpod_run(1)], {1: [job(verdict_job_name(PR, HEAD), conclusion)]}
                )
                self.assertIsNone(find_verdict(github, REPO, PR, HEAD))

    def test_only_a_dispatch_of_the_main_runpod_workflow_here_counts(self) -> None:
        for field, value in (
            ("path", ".github/workflows/CI_gpu.yml"),
            ("event", "pull_request"),
            ("head_branch", "feature"),
            ("repository", {"full_name": "someone/tenferro-rs"}),
        ):
            with self.subTest(field=field):
                github = FakeGitHub(
                    [runpod_run(1, **{field: value})],
                    {1: [job(verdict_job_name(PR, HEAD))]},
                )
                self.assertIsNone(find_verdict(github, REPO, PR, HEAD))


class WaitTests(unittest.TestCase):
    def test_waits_through_lookup_errors_until_the_deadline(self) -> None:
        now = [0.0]
        github = FakeGitHub([], fail=True)
        passed = wait_for_verdict(
            github,
            REPO,
            PR,
            HEAD,
            wait_minutes=2,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
        self.assertFalse(passed)
        self.assertEqual(now[0], 120.0)

    def test_a_verdict_ends_the_wait(self) -> None:
        github = FakeGitHub([runpod_run(1)], {1: [job(verdict_job_name(PR, HEAD))]})
        self.assertEqual(
            main(["wait", "--repository", REPO, "--pr", str(PR), "--head-sha", HEAD], github),
            0,
        )

    def test_an_abbreviated_head_is_refused(self) -> None:
        github = FakeGitHub([])
        self.assertEqual(
            main(["wait", "--repository", REPO, "--pr", str(PR), "--head-sha", "d143ce2e"], github),
            2,
        )


class RerunTests(unittest.TestCase):
    def gate_run(self, run_id: int, status: str, conclusion: str | None) -> dict:
        return {
            "id": run_id,
            "head_sha": HEAD,
            "status": status,
            "conclusion": conclusion,
            "created_at": f"2026-11-23T0{run_id}:00:00Z",
        }

    def test_the_newest_failed_gate_run_is_rerun(self) -> None:
        github = FakeGitHub(
            [],
            gate_runs=[self.gate_run(1, "completed", "success"), self.gate_run(2, "completed", "failure")],
        )
        self.assertIn("Re-ran", rerun_gate(github, REPO, PR, HEAD))
        self.assertEqual(
            github.posts,
            [f"https://api.github.com/repos/{REPO}/actions/runs/2/rerun-failed-jobs"],
        )

    def test_a_waiting_or_passed_gate_is_left_alone(self) -> None:
        for status, conclusion in (("in_progress", None), ("completed", "success")):
            with self.subTest(status=status):
                github = FakeGitHub([], gate_runs=[self.gate_run(1, status, conclusion)])
                rerun_gate(github, REPO, PR, HEAD)
                self.assertEqual(github.posts, [])

    def test_a_failed_rerun_never_fails_the_verdict(self) -> None:
        for github in (
            FakeGitHub([], fail=True),
            FakeGitHub([], gate_runs=[self.gate_run(1, "completed", "failure")], post_status=403),
        ):
            with self.subTest(github=github):
                code = main(
                    ["rerun", "--repository", REPO, "--pr", str(PR), "--head-sha", HEAD],
                    github,
                )
                self.assertEqual(code, 0)


class WorkflowWiringTests(unittest.TestCase):
    def test_the_verdict_job_name_matches_the_helper(self) -> None:
        workflow = (ROOT / ".github/workflows/runpod-gpu-test.yml").read_text()
        expected = verdict_job_name(
            "${{ needs.authorize.outputs.pr_number }}",  # type: ignore[arg-type]
            "${{ needs.authorize.outputs.target_head_sha }}",
        )
        self.assertIn(f'    name: "{expected}"\n', workflow)

    def test_the_gate_reads_the_helper_from_the_base_revision(self) -> None:
        workflow = (ROOT / ".github/workflows/CI_gpu.yml").read_text()
        checkout = workflow.split("      - name: Checkout the base revision's change policy\n", 1)[1]
        checkout = checkout.split("      - name:", 1)[0]
        self.assertIn("ref: ${{ github.event.pull_request.base.sha }}", checkout)
        self.assertIn("scripts/ci/fork_gpu_gate.py", checkout)

    def test_no_pull_request_reaches_the_ubuntu_gpu_runner(self) -> None:
        workflow = (ROOT / ".github/workflows/CI_gpu.yml").read_text()
        for name in ("pre-gpu-gate", "cuda-archive", "cuda-run"):
            header = workflow.split(f"\n  {name}:\n", 1)[1].split("    steps:\n", 1)[0]
            with self.subTest(job=name):
                self.assertIn("    if: github.event_name == 'workflow_dispatch'\n", header)


if __name__ == "__main__":
    unittest.main()
