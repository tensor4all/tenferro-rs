"""Run the RunPod `authorize` step for a maintainer's PR dispatch (#2717 et al.).

The secret-bearing workflow runs from `main`, so a PR's own CI never executes
its edits to it. These tests run the real `Check actor and PR source are
allowed` script with a stubbed `gh` that serves the PR and each login's
repository role, and assert whether a pod would be authorized.

Policy under test: a manual `pr_number` dispatch is approved by the
dispatching admin/maintain actor naming the reviewed head in
`approved_head_sha`. The PR author's role is not part of that decision.
"""

import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ".github/workflows/runpod-gpu-test.yml"
REPO = "tensor4all/tenferro-rs"
HEAD = "c" * 40
OTHER = "d" * 40

STUB = """\
gh() {
  printf '%s\\n' "$*" >> "${GH_LOG}"
  python3 "${GH_STUB}" "$@"
}
"""

GH_STUB = r'''
import json, os, sys
args = sys.argv[1:]
state = json.loads(open(os.environ["GH_STATE"]).read())
path = next(a for a in args if a.startswith("repos/"))
if "/collaborators/" in path:
    login = path.split("/collaborators/")[1].split("/")[0]
    print(state["roles"].get(login, "none"))
elif path.endswith("/labels"):
    pass
elif "/files" in path:
    print("\n".join(state["files"]))
elif "/pulls/" in path:
    # Each read serves the next snapshot; the last one repeats.
    index = state.get("reads", 0)
    snapshots = state["prs"]
    state["reads"] = index + 1
    open(os.environ["GH_STATE"], "w").write(json.dumps(state))
    print(json.dumps(snapshots[min(index, len(snapshots) - 1)]))
else:
    sys.exit(f"unexpected gh call: {args}")
'''


def authorize_script(workflow_text: str) -> str:
    body = workflow_text.split("      - name: Check actor and PR source are allowed\n", 1)[1]
    body = body.split("\n      - ", 1)[0]
    body = body.split("\n  runpod-contract:\n", 1)[0]
    return textwrap.dedent(body.split("        run: |\n", 1)[1])


def pr(head: str = HEAD, author: str = "frankebel", repo: str = REPO) -> dict:
    return {
        "state": "open",
        "user": {"login": author},
        "head": {"sha": head, "repo": {"full_name": repo}},
        "base": {"sha": "e" * 40},
    }


def dispatch(
    *,
    approved: str,
    roles: dict,
    prs: list,
    actor: str = "maintainer",
    ref: str = "refs/heads/main",
    workflow_text: str | None = None,
) -> tuple[subprocess.CompletedProcess, dict]:
    script = authorize_script(workflow_text or (ROOT / WORKFLOW).read_text())
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        script = script.replace("/tmp/", f"{directory}/")
        (tmp / "gh_stub.py").write_text(GH_STUB)
        (tmp / "state.json").write_text(
            json.dumps({"roles": roles, "prs": prs, "files": ["crates/tenferro-gpu/src/lib.rs"]})
        )
        output = tmp / "github_output"
        output.write_text("")
        env = dict(os.environ) | {
            "GH_TOKEN": "token",
            "GH_LOG": str(tmp / "gh.log"),
            "GH_STUB": str(tmp / "gh_stub.py"),
            "GH_STATE": str(tmp / "state.json"),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": REPO,
            "GITHUB_REF": ref,
            "GITHUB_ACTOR": actor,
            "EVENT_NAME": "workflow_dispatch",
            "MANUAL_PR_NUMBER": "2717",
            "MANUAL_TENFERRO_REF": "",
            "MANUAL_GATE_HEAD_SHA": "",
            "MANUAL_APPROVED_HEAD_SHA": approved,
            "WORKFLOW_RUN_ACTOR": "",
            "WORKFLOW_RUN_EVENT": "",
            "WORKFLOW_RUN_PULL_REQUESTS": "[]",
        }
        result = subprocess.run(
            ["bash", "-c", STUB + script],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        outputs = dict(
            line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
        )
    return result, outputs


class SameRepoApprovalTests(unittest.TestCase):
    roles = {"maintainer": "maintain", "frankebel": "write"}

    def test_write_role_author_is_validated_after_maintainer_approves_head(self) -> None:
        result, outputs = dispatch(approved=HEAD, roles=self.roles, prs=[pr()])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(outputs["fork_pr"], "false")
        self.assertEqual(outputs["tenferro_ref"], HEAD)
        self.assertEqual(outputs["target_head_sha"], HEAD)
        self.assertEqual(outputs["pr_number"], "2717")
        self.assertEqual(outputs["run_gpu"], "true")
        self.assertEqual(outputs["local_gpu_validation"], "false")

    def test_an_approved_head_that_is_not_the_pr_head_is_refused(self) -> None:
        result, outputs = dispatch(approved=OTHER, roles=self.roles, prs=[pr()])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"head is {HEAD}, not the approved {OTHER}", result.stdout)
        self.assertNotIn("tenferro_ref", outputs)

    def test_a_same_repo_dispatch_without_an_approved_head_is_refused(self) -> None:
        for approved in ("", HEAD[:12], HEAD.upper()):
            with self.subTest(approved=approved):
                result, outputs = dispatch(approved=approved, roles=self.roles, prs=[pr()])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pass its full head SHA as approved_head_sha", result.stdout)
                self.assertNotIn("tenferro_ref", outputs)

    def test_the_dispatcher_must_still_be_admin_or_maintain(self) -> None:
        result, outputs = dispatch(
            approved=HEAD, roles=self.roles, prs=[pr()], actor="frankebel"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Workflow actor frankebel has 'write' repository role", result.stdout)
        self.assertNotIn("tenferro_ref", outputs)

    def test_a_head_moved_during_authorization_is_refused(self) -> None:
        result, outputs = dispatch(
            approved=HEAD, roles=self.roles, prs=[pr(), pr(head=OTHER)]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("head moved during authorization", result.stdout)
        self.assertNotIn("tenferro_ref", outputs)

    def test_a_dispatch_from_a_non_main_workflow_ref_is_refused(self) -> None:
        result, _ = dispatch(
            approved=HEAD, roles=self.roles, prs=[pr()], ref="refs/heads/topic"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must use the trusted main workflow", result.stdout)


class ForkApprovalTests(unittest.TestCase):
    roles = {"maintainer": "maintain"}

    def test_an_approved_fork_head_is_authorized_as_a_fork(self) -> None:
        result, outputs = dispatch(
            approved=HEAD, roles=self.roles, prs=[pr(author="outsider", repo="outsider/tenferro-rs")]
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(outputs["fork_pr"], "true")
        self.assertEqual(outputs["target_head_sha"], HEAD)

    def test_a_fork_head_other_than_the_approved_one_is_refused(self) -> None:
        result, _ = dispatch(
            approved=OTHER, roles=self.roles, prs=[pr(author="outsider", repo="outsider/tenferro-rs")]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"Fork PR #2717 head is {HEAD}, not the approved {OTHER}", result.stdout)


if __name__ == "__main__":
    unittest.main()
