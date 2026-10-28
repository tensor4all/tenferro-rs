"""The local GPU gate publishes `CI GPU gate` only for a verified run on the head.

`scripts/ci/local_gpu_gate.py publish` posts the SHA-bound evidence and
dispatches `local-gpu-gate.yml`, whose `relay` step re-verifies everything before
the check is published (REPOSITORY_RULES.md, CI Cost Discipline). These tests pin
every refusal, the published payload, and the workflow's trust shape.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.ci import local_gpu_gate as gate
from scripts.ci.gpu_gate_reuse import MARKER_PREFIX as RUNPOD_MARKER

ROOT = Path(__file__).resolve().parents[3]
REPO = "tensor4all/tenferro-rs"
SHA = "a" * 40
OTHER = "b" * 40


def pr(**overrides) -> dict:
    value = {"state": "open", "head": {"sha": SHA, "repo": {"full_name": REPO}}}
    value.update(overrides)
    return value


def comment(**overrides) -> dict:
    value = {
        "id": 11,
        "issue_url": f"https://api.github.com/repos/{REPO}/issues/7",
        "html_url": "https://github.com/x/pull/7#issuecomment-11",
        "user": {"login": "maint"},
        "body": gate.marker_line(SHA) + "\nLocal GPU gate: **passed**\n",
    }
    value.update(overrides)
    return value


def relay_errors(**overrides) -> list[str]:
    arguments = {
        "ref": "refs/heads/main",
        "repository": REPO,
        "actor": "maint",
        "actor_role": "maintain",
        "head_sha": SHA,
        "pr": pr(),
        "labels": [gate.LABEL],
        "comment": comment(),
        "pr_number": 7,
    }
    arguments.update(overrides)
    return gate.relay_errors(**arguments)


class RelayVerificationTests(unittest.TestCase):
    def test_a_verified_run_on_the_head_is_accepted(self) -> None:
        self.assertEqual(relay_errors(), [])
        self.assertEqual(relay_errors(actor_role="admin"), [])

    def test_every_broken_premise_is_refused(self) -> None:
        cases = {
            "untrusted ref": {"ref": "refs/heads/feature"},
            "writer role": {"actor_role": "write"},
            "short sha": {"head_sha": SHA[:7]},
            "closed PR": {"pr": pr(state="closed")},
            "fork PR": {"pr": pr(head={"sha": SHA, "repo": {"full_name": "fork/x"}})},
            "pushed since the run": {"pr": pr(head={"sha": OTHER, "repo": {"full_name": REPO}})},
            "no label": {"labels": []},
            "comment on another PR": {
                "comment": comment(issue_url=f"https://api.github.com/repos/{REPO}/issues/8")
            },
            "comment by someone else": {"comment": comment(user={"login": "other"})},
            "evidence for an earlier head": {
                "comment": comment(body=gate.marker_line(OTHER) + "\n")
            },
            "a failed run": {
                "comment": comment(body=f"<!-- {gate.MARKER_PREFIX}:{SHA}:failed -->\n")
            },
            "marker not on the first line": {
                "comment": comment(body="quote:\n" + gate.marker_line(SHA) + "\n")
            },
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                self.assertNotEqual(relay_errors(**overrides), [])

    def test_the_check_is_bound_to_the_head_and_never_reusable(self) -> None:
        payload = gate.check_run_payload(
            head_sha=SHA, comment=comment(), run_url="https://x/actions/runs/5"
        )
        self.assertEqual(payload["name"], "CI GPU gate")
        self.assertEqual(payload["head_sha"], SHA)
        self.assertEqual(payload["conclusion"], "success")
        self.assertEqual(payload["external_id"], f"local-gpu-gate:v1:{SHA}")
        self.assertFalse(payload["external_id"].startswith(RUNPOD_MARKER))
        # Run-shaped, or CI_gpu.yml's poller would take it for its own job.
        self.assertNotIn("/job/", payload["details_url"])
        self.assertIn(comment()["html_url"], payload["output"]["summary"])
        self.assertIn("Local GPU gate", payload["output"]["summary"])


class FakeGitHub:
    """Answers `gh api` like GitHub, from mutable state, and records POSTs."""

    def __init__(self, *, reads_before_push: int | None = None) -> None:
        self.pr = pr()
        self.labels = [{"name": gate.LABEL}]
        self.comment = comment()
        self.role = "maintain"
        self.posted: list[dict] = []
        self.reads = 0
        self.reads_before_push = reads_before_push

    def __call__(self, command, **kwargs) -> subprocess.CompletedProcess[str]:
        path = command[4]
        if path.endswith("/pulls/7"):
            self.reads += 1
            if self.reads_before_push is not None and self.reads > self.reads_before_push:
                self.pr = pr(head={"sha": OTHER, "repo": {"full_name": REPO}})
            data = self.pr
        elif path.endswith("/labels"):
            data = self.labels
        elif "/issues/comments/" in path:
            data = self.comment
        elif "/permission" in path:
            data = {"role_name": self.role}
        elif path.endswith("/check-runs"):
            self.posted.append(json.loads(kwargs["input"]))
            data = {}
        else:
            raise AssertionError(f"unexpected gh call: {command}")
        return subprocess.CompletedProcess(command, 0, json.dumps(data), "")


RELAY_ENV = {
    "GITHUB_REPOSITORY": REPO,
    "GITHUB_ACTOR": "maint",
    "GITHUB_REF": "refs/heads/main",
    "PR_NUMBER": "7",
    "HEAD_SHA": SHA,
    "COMMENT_ID": "11",
    "RUN_URL": "https://x/actions/runs/5",
}


class RelayTests(unittest.TestCase):
    def test_publishes_once_after_verifying_twice(self) -> None:
        github = FakeGitHub()
        self.assertEqual(gate.relay(RELAY_ENV, runner=github), 0)
        self.assertEqual(len(github.posted), 1)
        self.assertEqual(github.posted[0]["head_sha"], SHA)
        self.assertEqual(github.reads, 2)

    def test_a_push_during_the_relay_publishes_nothing(self) -> None:
        github = FakeGitHub(reads_before_push=1)
        self.assertEqual(gate.relay(RELAY_ENV, runner=github), 1)
        self.assertEqual(github.posted, [])

    def test_a_non_maintainer_dispatch_publishes_nothing(self) -> None:
        github = FakeGitHub()
        github.role = "write"
        self.assertEqual(gate.relay(RELAY_ENV, runner=github), 1)
        self.assertEqual(github.posted, [])


class PublishTests(unittest.TestCase):
    def errors(self, **overrides) -> list[str]:
        arguments = {
            "record": {"sha": SHA, "passed": True},
            "pr": pr(),
            "labels": [gate.LABEL],
            "role": "admin",
            "login": "maint",
            "repository": REPO,
        }
        arguments.update(overrides)
        return gate.publish_errors(**arguments)

    def test_a_passing_record_for_the_head_is_accepted(self) -> None:
        self.assertEqual(self.errors(), [])

    def test_refusals(self) -> None:
        cases = {
            "writer": {"role": "write"},
            "failed run": {"record": {"sha": SHA, "passed": False}},
            "stale head": {"record": {"sha": OTHER, "passed": True}},
            "unlabelled": {"labels": []},
            "fork": {"pr": pr(head={"sha": SHA, "repo": {"full_name": "fork/x"}})},
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                self.assertNotEqual(self.errors(**overrides), [])

    def test_nextest_summaries_are_extracted(self) -> None:
        line = "Summary [  12.3s] 181 tests run: 181 passed, 3 skipped"
        for output in (
            f"noise\n     {line}\nend\n",
            f"noise\n\x1b[32;1m     Summary\x1b[0m [  12.3s] 181 tests run: 181 passed, 3 skipped\n",
        ):
            with self.subTest(output=output):
                self.assertEqual(gate.nextest_summaries(output), [line])


class PublishFlowTests(unittest.TestCase):
    """`publish` posts the record and dispatches the relay, or touches nothing."""

    def publish(self, *, head: str) -> tuple[int, list[list[str]], list[dict]]:
        calls: list[list[str]] = []
        posted: list[dict] = []

        def runner(command, **kwargs):
            calls.append(list(command))
            if command[:2] == ["gh", "workflow"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            path = command[4]
            if path == "user":
                data = {"login": "maint"}
            elif path.endswith("/pulls/7"):
                data = pr(head={"sha": head, "repo": {"full_name": REPO}})
            elif path.endswith("/labels"):
                data = [{"name": gate.LABEL}]
            elif "/permission" in path:
                data = {"role_name": "admin"}
            elif path.endswith("/issues/7/comments"):
                posted.append(json.loads(kwargs["input"]))
                data = {"id": 99, "html_url": "https://x/c99"}
            else:
                raise AssertionError(f"unexpected call {command}")
            return subprocess.CompletedProcess(command, 0, json.dumps(data), "")

        with tempfile.TemporaryDirectory() as directory:
            record = {"sha": SHA, "passed": True, "body": gate.marker_line(SHA) + "\n"}
            Path(directory, f"{SHA}.json").write_text(json.dumps(record))
            with mock.patch.object(gate, "evidence_dir", return_value=Path(directory)):
                args = argparse.Namespace(repo=REPO, sha=SHA, pr=7)
                code = gate.publish(args, runner=runner)
        return code, calls, posted

    def test_posts_the_record_then_dispatches_the_relay_on_main(self) -> None:
        code, calls, posted = self.publish(head=SHA)
        self.assertEqual(code, 0)
        self.assertEqual(posted, [{"body": gate.marker_line(SHA) + "\n"}])
        self.assertEqual(
            calls[-1],
            ["gh", "workflow", "run", "local-gpu-gate.yml", "--repo", REPO, "--ref",
             "main", "-f", "pr_number=7", "-f", f"head_sha={SHA}", "-f", "comment_id=99"],
        )

    def test_a_moved_head_posts_and_dispatches_nothing(self) -> None:
        code, calls, posted = self.publish(head=OTHER)
        self.assertEqual(code, 1)
        self.assertEqual(posted, [])
        self.assertFalse(any(command[:2] == ["gh", "workflow"] for command in calls))


class LaneStepTests(unittest.TestCase):
    def test_every_lane_of_the_pod_runs_locally(self) -> None:
        steps = gate.lane_steps(
            {"BASE": "1"}, {"PJRT": "1"}, 4, Path("/t"), Path("/o")
        )
        names = [name for name, *_ in steps]
        self.assertEqual(
            names,
            ["CUDA archive", "PJRT archive", "CUDA tutorial build", "CUDA host lane",
             "PJRT host lane", "CUDA device lane", "CUDA tutorial", "PJRT device lane"],
        )
        by_name = {name: (command, env) for name, command, env in steps}
        cuda_archive = by_name["CUDA archive"][0]
        for crate in gate.CUDA_CRATES:
            self.assertIn(crate, cuda_archive)
        self.assertEqual(cuda_archive[cuda_archive.index("--features") + 1], "cuda")
        self.assertEqual(by_name["CUDA host lane"][1]["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn("gpu", by_name["CUDA device lane"][0])
        self.assertEqual(by_name["CUDA tutorial"][0], ["/t/ci/cuda_tutorial"])
        self.assertEqual(by_name["CUDA tutorial"][1]["TENFERRO_REQUIRE_CUDA"], "1")
        self.assertEqual(by_name["PJRT device lane"][1], {"PJRT": "1"})


class WorkflowShapeTests(unittest.TestCase):
    def test_the_relay_holds_no_secret_and_runs_only_on_dispatch(self) -> None:
        text = (ROOT / ".github/workflows/local-gpu-gate.yml").read_text()
        self.assertIn("on:\n  workflow_dispatch:\n", text)
        self.assertNotIn("pull_request", text)
        self.assertNotIn("workflow_run", text)
        self.assertNotIn("secrets.", text)
        self.assertNotIn("RUNPOD", text)
        self.assertIn("checks: write", text)
        self.assertIn("python3 scripts/ci/local_gpu_gate.py relay", text)
        # A branch copy of the relay must never run with `checks: write`.
        self.assertIn("if: github.ref == 'refs/heads/main'", text)
        self.assertIn("ref: main", text)

    def test_runpod_skips_every_job_for_a_labelled_pr(self) -> None:
        text = (ROOT / ".github/workflows/runpod-gpu-test.yml").read_text()
        skip = (
            "if: needs.authorize.outputs.gpu_required == 'true' && "
            "needs.authorize.outputs.local_gpu_validation != 'true'"
        )
        self.assertEqual(text.count(skip), 4)
        self.assertNotIn("if: needs.authorize.outputs.gpu_required == 'true'\n", text)
        self.assertNotIn("Local GPU validation:", text)

    def test_the_old_comment_route_is_gone(self) -> None:
        for workflow in ("CI_gpu.yml", "runpod-gpu-test.yml"):
            with self.subTest(workflow):
                text = (ROOT / ".github/workflows" / workflow).read_text()
                self.assertNotIn("Local GPU validation:", text)


if __name__ == "__main__":
    unittest.main()
