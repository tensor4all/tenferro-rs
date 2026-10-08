"""Workflow contracts for the paid-GPU cost controls of #2002.

The RunPod workflows run from the default branch (`workflow_run`), so a PR's
own CI never executes its edits to them. These tests run the real step
scripts with stubbed `gh` calls instead, so every new branch is exercised
before merge.
"""

import json
import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.ci.gpu_gate_reuse import gate_external_id

ROOT = Path(__file__).resolve().parents[3]
PARENT = ".github/workflows/runpod-gpu-test.yml"
CHILD = ".github/workflows/runpod-gpu-execute.yml"
HEAD = "a" * 40
REF = "b" * 40


def text(path: str) -> str:
    return (ROOT / path).read_text()


def job(workflow: str, name: str) -> str:
    body = workflow.split(f"\n  {name}:\n", 1)[1]
    match = re.search(r"\n  [a-z][a-z0-9-]*:\n", body)
    return body[: match.start()] if match else body


def steps(job_text: str) -> list[str]:
    return job_text.split("\n    steps:\n", 1)[1].split("\n      - ")[1:]


def step(workflow: str, name: str) -> str:
    body = workflow.split(f"      - name: {name}\n", 1)[1]
    return body.split("\n      - ", 1)[0]


def step_script(workflow: str, name: str) -> str:
    return textwrap.dedent(step(workflow, name).split("        run: |\n", 1)[1])


class GateReuseDecisionTests(unittest.TestCase):
    """`Decide whether the paid path runs` with the reuse lookup (item 1)."""

    def decide(self, *, checks, files=("crates/tenferro-gpu/src/lib.rs",), labels=(),
               pr_number="7", state="open", break_lookup=False) -> dict:
        script = step_script(text(CHILD), "Decide whether the paid path runs")
        with tempfile.TemporaryDirectory() as directory:
            checks_file = Path(directory) / "checks.json"
            checks_file.write_text(json.dumps({"check_runs": checks}))
            output = Path(directory) / "github_output"
            output.write_text("")
            lookup = "python3 scripts/ci/gpu_gate_reuse.py \\\n"
            self.assertIn(lookup, script)
            replacement = (
                "python3 scripts/ci/does_not_exist.py \\\n" if break_lookup
                else f"python3 scripts/ci/gpu_gate_reuse.py --checks-json {checks_file} \\\n"
            )
            script = script.replace("/tmp/", f"{directory}/")
            script = script.replace(lookup, replacement)
            stub = textwrap.dedent(
                """\
                gh() {
                  case "$*" in
                    *pulls/*/files*) printf '%s\\n' $FILES ;;
                    *issues/*/labels*) printf '%s\\n' $LABELS ;;
                    *pulls/*) printf '%s\\n' "$PR_STATE" ;;
                    *) echo "unexpected gh call: $*" >&2; return 1 ;;
                  esac
                }
                """
            )
            env = dict(
                os.environ,
                GITHUB_REPOSITORY="tensor4all/tenferro-rs",
                GITHUB_OUTPUT=str(output),
                PR_NUMBER=pr_number,
                TARGET_HEAD_SHA=HEAD,
                TESTED_REF=REF,
                FILES=" ".join(files),
                LABELS=" ".join(labels),
                PR_STATE=state,
            )
            env.pop("GH_TOKEN", None)
            result = subprocess.run(
                ["bash", "-c", stub + script], cwd=ROOT, env=env,
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            values = {}
            for line in output.read_text().splitlines():
                key, _, value = line.partition("=")
                values[key] = value
            return values

    @staticmethod
    def gate(kind: str, conclusion: str = "success", ref: str = REF, check_id: int = 1,
             completed_at: str = "2026-10-04T10:00:00Z") -> dict:
        return {
            "id": check_id, "name": "CI GPU gate", "status": "completed",
            "conclusion": conclusion, "completed_at": completed_at,
            "external_id": gate_external_id(ref, kind), "app": {"slug": "github-actions"},
            "html_url": f"https://github.com/tensor4all/tenferro-rs/runs/{check_id}",
        }

    def test_paid_success_for_the_same_ref_is_reused(self) -> None:
        values = self.decide(checks=[self.gate("paid")])
        self.assertEqual(values["run_paid_path"], "false")
        self.assertEqual(values["decision"], "reuse")
        self.assertEqual(values["reused_check_url"], "https://github.com/tensor4all/tenferro-rs/runs/1")

    def test_anything_else_runs_the_paid_path(self) -> None:
        later_failure = self.gate("failed", "failure", check_id=2, completed_at="2026-10-04T11:00:00Z")
        for name, checks in (
            ("no gate", []),
            ("other ref", [self.gate("paid", ref="c" * 40)]),
            ("newer failure", [self.gate("paid"), later_failure]),
            ("not required", [self.gate("not-required")]),
            ("local", [self.gate("local")]),
        ):
            with self.subTest(name=name):
                values = self.decide(checks=checks)
                self.assertEqual(values["run_paid_path"], "true")
                self.assertEqual(values["decision"], "run")

    def test_a_broken_lookup_fails_open_to_running(self) -> None:
        values = self.decide(checks=[self.gate("paid")], break_lookup=True)
        self.assertEqual(values["run_paid_path"], "true")
        self.assertEqual(values["decision"], "run")

    def test_existing_skips_are_unchanged(self) -> None:
        for kwargs in (
            {"state": "closed"},
            {"labels": ("gpu-validated-locally",)},
            {"files": ("docs/guides/devices-and-gpu.md",)},
        ):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                values = self.decide(checks=[], **kwargs)
                self.assertEqual(values["run_paid_path"], "false")
                self.assertEqual(values["decision"], "skip")
        values = self.decide(checks=[self.gate("paid")], pr_number="0")
        self.assertEqual((values["run_paid_path"], values["decision"]), ("true", "run"))

    def test_decision_reaches_the_gate(self) -> None:
        child = text(CHILD)
        self.assertIn("paid_path_decision: ${{ steps.paid_path.outputs.decision }}", child)
        self.assertIn("value: ${{ jobs.start-runpod.outputs.paid_path_decision }}", child)
        self.assertIn("value: ${{ jobs.start-runpod.outputs.reused_check_url }}", child)
        parent = text(PARENT)
        self.assertIn("PAID_PATH_DECISION: ${{ needs.gpu-execution.outputs.paid_path_decision }}", parent)
        self.assertIn("TESTED_REF: ${{ needs.authorize.outputs.tenferro_ref }}", parent)


class GatePublicationTests(unittest.TestCase):
    """`Publish required PR check` records the tested ref and gate kind."""

    def publish(self, **overrides) -> tuple[int, dict]:
        script = step_script(text(PARENT), "Publish required PR check")
        with tempfile.TemporaryDirectory() as directory:
            check_file = f"{directory}/check.json"
            script = script.replace("/tmp/ci-gpu-gate-check.json", check_file)
            stub = textwrap.dedent(
                """\
                gh() {
                  case "$*" in
                    *--method\\ POST*) return 0 ;;
                    *comments*) printf 'maint\\thttps://example.invalid/c1\\n' ;;
                    *permission*) echo maintain ;;
                    *) echo "unexpected gh call: $*" >&2; return 1 ;;
                  esac
                }
                """
            )
            env = dict(
                os.environ,
                GITHUB_REPOSITORY="tensor4all/tenferro-rs",
                GITHUB_RUN_ID="123",
                TARGET_HEAD_SHA=HEAD,
                TESTED_REF=REF,
                RUN_URL="https://example.invalid/run",
                AUTHORIZE_RESULT="success",
                GPU_REQUIRED="true",
                LOCAL_GPU_VALIDATION="false",
                PR_NUMBER="7",
                POLICY_REASON="code",
                CONTRACT_RESULT="success",
                PRE_RUNPOD_RESULT="success",
                ARCHIVE_RESULT="success",
                RUNTIME_RESULT="success",
                GPU_EXECUTION_RESULT="success",
                PAID_PATH_DECISION="run",
                REUSED_CHECK_URL="",
            )
            env.update(overrides)
            result = subprocess.run(["bash", "-c", stub + script], env=env,
                                    capture_output=True, text=True)
            payload = json.loads(Path(check_file).read_text())
            return result.returncode, payload

    def test_paid_success_is_marked_reusable(self) -> None:
        status, check = self.publish()
        self.assertEqual(status, 0)
        self.assertEqual(check["conclusion"], "success")
        self.assertEqual(check["external_id"], gate_external_id(REF, "paid"))

    def test_reuse_links_the_earlier_gate(self) -> None:
        status, check = self.publish(PAID_PATH_DECISION="reuse", REUSED_CHECK_URL="https://x/runs/1")
        self.assertEqual(status, 0)
        self.assertEqual(check["external_id"], gate_external_id(REF, "reused"))
        self.assertEqual(check["output"]["title"], "RunPod CI GPU gate passed (reused)")
        self.assertIn("https://x/runs/1", check["output"]["summary"])

    def test_reuse_without_a_source_fails(self) -> None:
        status, check = self.publish(PAID_PATH_DECISION="reuse", REUSED_CHECK_URL="")
        self.assertEqual(status, 1)
        self.assertEqual(check["conclusion"], "failure")
        self.assertEqual(check["external_id"], gate_external_id(REF, "failed"))

    def test_failures_and_non_paid_successes_are_not_reusable(self) -> None:
        for overrides, conclusion, kind in (
            ({"GPU_EXECUTION_RESULT": "failure"}, "failure", "failed"),
            ({"RUNTIME_RESULT": "failure"}, "failure", "failed"),
            ({"GPU_REQUIRED": "false"}, "success", "not-required"),
            ({"PAID_PATH_DECISION": "skip"}, "success", "skipped"),
            ({"PAID_PATH_DECISION": ""}, "success", "skipped"),
            ({"LOCAL_GPU_VALIDATION": "true", "GPU_EXECUTION_RESULT": "skipped",
              "PAID_PATH_DECISION": "skip"}, "success", "local"),
        ):
            with self.subTest(**overrides):
                _, check = self.publish(**overrides)
                self.assertEqual(check["conclusion"], conclusion)
                self.assertEqual(check["external_id"], gate_external_id(REF, kind))

    def test_no_tested_ref_publishes_no_marker(self) -> None:
        status, check = self.publish(TESTED_REF="")
        self.assertEqual(status, 0)
        self.assertNotIn("external_id", check)

    def test_failure_summary_names_the_retry(self) -> None:
        _, check = self.publish(GPU_EXECUTION_RESULT="failure")
        self.assertIn("gh run rerun 123 --failed", check["output"]["summary"])

    def test_local_label_reaches_the_gate(self) -> None:
        # The label decision used to be printed to stdout, so the gate never
        # verified the evidence comment of a labelled PR.
        authorize = step_script(text(PARENT), "Check actor and PR source are allowed")
        self.assertIn('echo "local_gpu_validation=true" >> "${GITHUB_OUTPUT}"', authorize)
        self.assertIn('echo "local_gpu_validation=false" >> "${GITHUB_OUTPUT}"', authorize)
        self.assertNotRegex(authorize, r'echo "local_gpu_validation=(true|false)"\n')


class SetupBoundTests(unittest.TestCase):
    """Every step before the tests on the pod is time-bounded (item 2)."""

    SETUP_BOUND_MINUTES = 40

    def test_every_pre_test_step_is_bounded(self) -> None:
        gpu = job(text(CHILD), "run-gpu-tests")
        total = 0
        for block in steps(gpu):
            if block.startswith("name: Run CUDA tests from archive"):
                break
            bound = re.search(r"\n        timeout-minutes: (\d+)\n", "\n" + block)
            self.assertIsNotNone(bound, block.splitlines()[0])
            assert bound is not None
            total += int(bound.group(1))
        else:
            self.fail("test step not found")
        self.assertLessEqual(total, self.SETUP_BOUND_MINUTES)
        job_timeout = int(re.search(r"\n    timeout-minutes: (\d+)\n", gpu).group(1))
        self.assertLess(total, job_timeout)

    def test_cache_restores_abort_stalls_and_fall_back(self) -> None:
        gpu = job(text(CHILD), "run-gpu-tests")
        restores = [block for block in steps(gpu) if "actions/cache/restore@" in block]
        self.assertEqual(len(restores), 1)
        hosted = job(text(".github/workflows/runpod-gpu-runtime.yml"), "prepare-runtime")
        restores += [block for block in steps(hosted) if "actions/cache/restore@" in block]
        self.assertEqual(len(restores), 4)
        for block in restores:
            self.assertIn('SEGMENT_DOWNLOAD_TIMEOUT_MINS: "2"', block)
            self.assertIn("continue-on-error: true", block)

    def test_artifact_download_is_retried_once(self) -> None:
        gpu = job(text(CHILD), "run-gpu-tests")
        downloads = [block for block in steps(gpu) if "actions/download-artifact@" in block
                     and "inputs.archive_artifact_name" in block]
        self.assertEqual(len(downloads), 2)
        self.assertIn("continue-on-error: true", downloads[0])
        self.assertIn("steps.archive_download.outcome != 'success'", downloads[1])
        self.assertNotIn("continue-on-error", downloads[1])


class TelemetryTests(unittest.TestCase):
    """Cost telemetry and provision log timing (item 3)."""

    def test_provisioner_is_unbuffered(self) -> None:
        provision = step(text(CHILD), "Provision cheapest compatible RunPod pod")
        self.assertIn('PYTHONUNBUFFERED: "1"', provision)

    def test_cost_report_never_blocks_deletion(self) -> None:
        cleanup = job(text(CHILD), "cleanup-runpod")
        blocks = steps(cleanup)
        names = [block.splitlines()[0] for block in blocks]
        self.assertEqual(
            names,
            [
                "name: Checkout trusted RunPod cost report",
                "name: Report RunPod paid time and estimated cost",
                "name: Delete RunPod pod",
                "name: Report paid GPU CI cost by stage",
                "name: Save paid GPU CI cost record",
            ],
        )
        for block in blocks[:2]:
            self.assertIn("continue-on-error: true", block)
        self.assertIn("python3 scripts/ci/runpod_cost.py", blocks[1])
        self.assertNotIn("continue-on-error", blocks[2])
        self.assertNotIn("runpod_cost", blocks[2])

    def test_stage_report_runs_only_after_confirmed_deletion(self) -> None:
        child = text(CHILD)
        cleanup = job(child, "cleanup-runpod")
        self.assertLess(cleanup.index("name: Delete RunPod pod"), cleanup.index("name: Report paid GPU CI cost by stage"))
        self.assertIn("actions: read", cleanup)
        for name in ("Report paid GPU CI cost by stage", "Save paid GPU CI cost record"):
            block = step(child, name)
            self.assertIn("continue-on-error: true", block)
            self.assertIn("steps.delete_pod.outputs.deleted_at != ''", block)
        deletion = step(child, "Delete RunPod pod")
        self.assertLess(deletion.index("Failed to delete RunPod pod"), deletion.index("echo \"deleted_at="))


class RunnerPinCheckWorkflowTests(unittest.TestCase):
    """Free scheduled detection of a stale runner pin (item 4)."""

    def test_scheduled_read_only_and_exercised_by_pin_changes(self) -> None:
        workflow = text(".github/workflows/runner-pin-check.yml")
        self.assertIn("  schedule:\n", workflow)
        self.assertIn("  workflow_dispatch:\n", workflow)
        self.assertIn("      - .github/workflows/runpod-gpu-execute.yml\n", workflow)
        self.assertIn("      - scripts/ci/runner_pin_check.py\n", workflow)
        self.assertIn("permissions:\n  contents: read\n\n", workflow)
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("write", workflow)
        self.assertIn("python3 scripts/ci/runner_pin_check.py", workflow)

    def test_runbook_is_linked_from_the_pin(self) -> None:
        child = text(CHILD)
        pin = child[child.index("# Keep this pin current") : child.index('RUNNER_VERSION="')]
        self.assertIn("runner-pin-check.yml", pin)
        self.assertIn("Runner pin runbook", pin)
        self.assertIn("## Runner pin runbook", text("docs/design/runpod-gpu-provisioning.md"))


if __name__ == "__main__":
    unittest.main()
