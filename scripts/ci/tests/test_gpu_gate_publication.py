"""Run the CI GPU gate publication step and assert what it decides (#2391)."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[3]

STUBS = """\
gh() { printf '%s\\n' "$*" >> "$GH_LOG"; }
jq() {
  python3 - "$@" <<'PY'
import json, sys
args = sys.argv[1:]
pairs = {}
index = 0
while index < len(args):
    if args[index] == "--arg":
        pairs[args[index + 1]] = args[index + 2]
        index += 3
    else:
        index += 1
print(json.dumps(pairs))
PY
}
"""


def publish_script() -> str:
    text = (ROOT / ".github/workflows/runpod-gpu-test.yml").read_text()
    step = text.split("      - name: Publish required PR check\n", 1)[1]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


class GpuGatePublicationTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "GH_TOKEN": "token",
            "EVENT_NAME": "workflow_run",
            "TRIGGER_EVENT": "pull_request",
            "TARGET_HEAD_SHA": "",
            "TESTED_REF": "",
            "RUN_URL": "https://example.invalid/run",
            "AUTHORIZE_RESULT": "failure",
            "GPU_REQUIRED": "",
            "LOCAL_GPU_VALIDATION": "false",
            "PR_NUMBER": "",
            "POLICY_REASON": "",
            "CONTRACT_RESULT": "skipped",
            "PRE_RUNPOD_RESULT": "skipped",
            "ARCHIVE_RESULT": "skipped",
            "RUNTIME_RESULT": "skipped",
            "GPU_EXECUTION_RESULT": "skipped",
            "PAID_PATH_DECISION": "",
            "REUSED_CHECK_URL": "",
            "GITHUB_REPOSITORY": "tensor4all/tenferro-rs",
            "GITHUB_RUN_ID": "1",
        }

    def publish(self, **overrides):
        env = dict(os.environ) | self.base | overrides
        with tempfile.TemporaryDirectory() as directory:
            env["GH_LOG"] = f"{directory}/gh.log"
            payload = f"{directory}/check.json"
            result = subprocess.run(
                ["bash", "-c", STUBS + publish_script().replace(
                    "/tmp/ci-gpu-gate-check.json", payload
                )],
                env=env, capture_output=True, text=True,
            )
            log = Path(env["GH_LOG"])
            posted = log.read_text() if log.exists() else ""
            check = Path(payload).read_text() if Path(payload).exists() else ""
        return result, posted, check

    def test_push_without_a_pr_head_skips_and_succeeds(self):
        """A `main` push dispatches the GPU test but has no PR check to publish."""

        result, posted, _ = self.publish(TRIGGER_EVENT="push")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("skipping PR-head CI GPU gate publication", result.stdout)
        self.assertIn("trigger: push", result.stdout)
        self.assertNotIn("check-runs", posted)
        # Nothing to publish means no policy is required from authorize.
        self.assertNotIn("::error::", result.stdout)

    def test_missing_authorize_policy_at_a_pr_head_is_still_an_error(self):
        result, posted, check = self.publish(
            TARGET_HEAD_SHA="a" * 40, PR_NUMBER="7", TESTED_REF="a" * 40
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("Authorize job did not publish a valid GPU policy", result.stdout)
        self.assertIn("check-runs", posted)
        self.assertEqual(json.loads(check)["conclusion"], "failure")

    def test_a_pr_head_with_a_valid_policy_publishes(self):
        result, posted, check = self.publish(
            TARGET_HEAD_SHA="b" * 40,
            PR_NUMBER="7",
            TESTED_REF="b" * 40,
            AUTHORIZE_RESULT="success",
            GPU_REQUIRED="true",
            CONTRACT_RESULT="success",
            PRE_RUNPOD_RESULT="success",
            ARCHIVE_RESULT="success",
            RUNTIME_RESULT="success",
            GPU_EXECUTION_RESULT="success",
            PAID_PATH_DECISION="run",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Publishing CI GPU gate check to PR head SHA", result.stdout)
        payload = json.loads(check)
        self.assertEqual(payload["conclusion"], "success")
        self.assertEqual(payload["head_sha"], "b" * 40)
        self.assertEqual(payload["external_id"], f"runpod-gpu-gate:v1:{'b' * 40}:paid")


if __name__ == "__main__":
    unittest.main()
