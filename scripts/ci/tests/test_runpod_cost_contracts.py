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
        deletion = cleanup.index("      - name: Delete RunPod pod")
        self.assertLess(deletion, cleanup.index("name: Checkout trusted RunPod cost report"))
        self.assertLess(deletion, cleanup.index("name: Report paid GPU CI cost by stage"))
        self.assertIn("--record-output /tmp/runpod-pod.json", step(text(CHILD), "Delete RunPod pod"))
        self.assertIn("ref: ${{ github.workflow_sha }}", cleanup[:deletion])
        self.assertIn("persist-credentials: false", cleanup[:deletion])
        self.assertNotIn("continue-on-error", step(text(CHILD), "Delete RunPod pod"))
        checkout = step(text(CHILD), "Checkout trusted RunPod cost report")
        self.assertIn("steps.delete_pod.outputs.deleted_at != ''", checkout)
        self.assertIn("continue-on-error: true", checkout)

    def test_startup_snapshot_reaches_report_without_delaying_cleanup(self) -> None:
        workflow = text(CHILD)
        capture = step(workflow, "Preserve accepted Pod metadata")
        self.assertIn("continue-on-error: true", capture)
        self.assertIn("scripts.ci.runpod_metadata", capture)
        self.assertIn("--creation-record /tmp/runpod-response.json", capture)
        self.assertIn("pod_metadata: ${{ steps.pod_metadata.outputs.pod_metadata }}", workflow)
        reporting = step(workflow, "Report paid GPU CI cost by stage")
        self.assertIn("RUNPOD_STARTUP_METADATA: ${{ needs.start-runpod.outputs.pod_metadata }}", reporting)
        self.assertIn('--fallback-pod-json "${RUNPOD_STARTUP_METADATA}"', reporting)
        self.assertIn('--pod-id "${POD_ID}"', reporting)

    def test_failed_cost_record_read_still_deletes_pod(self) -> None:
        from scripts.ci.runpod_http import Response, delete_pod
        from scripts.ci.tests.test_runpod_http import Clock
        for status in (200, 0, 500):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                clock = Clock()
                calls = []
                def transport(request, *, timeout):
                    calls.append((request.method, clock.now, timeout))
                    return Response(status, {}, b"{}") if request.method == "GET" else Response(204, {}, b"")
                self.assertTrue(delete_pod("https://fixture.invalid/v2/pods", "key", "pod",
                    record_path=Path(directory)/"record.json", transport=transport,
                    monotonic=clock.monotonic, sleep=clock.sleep))
                self.assertEqual([call[0] for call in calls], ["GET", "DELETE"])
                self.assertEqual(calls[0][2], 5)
        script = step_script(text(CHILD), "Delete RunPod pod")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"output"
            stub = 'python3() { return "$HELPER_STATUS"; }\n'
            for status in (0, 1):
                output.unlink(missing_ok=True)
                run = subprocess.run(["bash", "-c", stub + script], capture_output=True, text=True,
                    env={**os.environ, "POD_ID":"pod", "GITHUB_OUTPUT":str(output), "HELPER_STATUS":str(status)})
                self.assertEqual(run.returncode == 0, status == 0)
                self.assertEqual(output.exists(), status == 0)

    def test_called_cleanup_read_permissions_are_granted_by_caller(self) -> None:
        caller = job(text(PARENT), "gpu-execution")
        cleanup = job(text(CHILD), "cleanup-runpod")
        self.assertIn("      actions: read", cleanup)
        # Reusable workflows cannot elevate permissions above the caller.
        # An explicit block must retain checks/contents needed by other jobs.
        permissions = re.search(r"    permissions:\n((?:      [^\n]+\n)+)", caller)
        self.assertIsNotNone(permissions, "GPU caller must grant cleanup's actions: read")
        scopes = dict(re.findall(r"      (\S+): (\S+)", permissions.group(1)))
        self.assertEqual(scopes, {"checks": "read", "contents": "read", "actions": "write"})
        self.assertNotIn("actions: write", job(text(CHILD), "run-gpu-tests"))

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
        self.assertLess(deletion.index("Could not confirm deletion of RunPod pod"), deletion.index("echo \"deleted_at="))


class RunnerImageTests(unittest.TestCase):
    """Resolve the current official image before spending on a Pod."""

    def resolve(self, manifest: dict, *, registry_failure: bool = False):
        script = step_script(text(CHILD), "Resolve latest official runner image")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / "docker"
            docker.write_text('''#!/bin/sh
printf '%s\\n' "$@" > "$DOCKER_ARGS"
if [ "$REGISTRY_FAILURE" = true ]; then exit 1; fi
printf '%s\\n' "$MANIFEST"
''')
            docker.chmod(0o755)
            output, summary, arguments = (root / name for name in ("output", "summary", "args"))
            env = dict(os.environ, PATH=f'{root}:{os.environ["PATH"]}',
                       RUNPOD_IMAGE="ghcr.io/actions/actions-runner:latest",
                       GITHUB_OUTPUT=str(output), GITHUB_STEP_SUMMARY=str(summary),
                       DOCKER_ARGS=str(arguments), MANIFEST=json.dumps(manifest),
                       REGISTRY_FAILURE=str(registry_failure).lower())
            result = subprocess.run(["bash", "-c", script], env=env,
                                    capture_output=True, text=True)
            return (result, output.read_text() if output.exists() else "",
                    summary.read_text() if summary.exists() else "",
                    arguments.read_text().splitlines())

    def test_each_run_resolves_latest_and_records_its_exact_image(self) -> None:
        for digest in ("sha256:" + "a" * 64, "sha256:" + "b" * 64):
            with self.subTest(digest=digest):
                result, output, summary, arguments = self.resolve({"digest": digest})
                self.assertEqual(result.returncode, 0, result.stderr)
                image = f"ghcr.io/actions/actions-runner:latest@{digest}"
                self.assertEqual(output, f"image={image}\n")
                self.assertIn(image, summary)
                self.assertIn(image, result.stdout)
                self.assertEqual(arguments, ["buildx", "imagetools", "inspect",
                    "ghcr.io/actions/actions-runner:latest", "--format", "{{json .Manifest}}"])

    def test_lookup_failures_do_not_publish_an_image(self) -> None:
        for manifest, failed in (({}, False), ({"digest": "unused"}, True)):
            with self.subTest(manifest=manifest, registry_failure=failed):
                result, output, summary, _ = self.resolve(manifest, registry_failure=failed)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")
                self.assertEqual(summary, "")

    def test_resolution_precedes_provisioning_and_uses_the_resolved_image(self) -> None:
        child = text(CHILD)
        resolve = step(child, "Resolve latest official runner image")
        self.assertIn("if: steps.paid_path.outputs.run_paid_path != 'false'", resolve)
        self.assertNotIn("secrets.", resolve)
        self.assertLess(child.index("name: Resolve latest official runner image"),
                        child.index("name: Create GitHub App token"))
        provision = step(child, "Provision cheapest compatible RunPod pod")
        self.assertIn("RUNPOD_IMAGE: ${{ steps.runner_image.outputs.image }}", provision)
        self.assertIn('--image-name "${RUNPOD_IMAGE}"', provision)


if __name__ == "__main__":
    unittest.main()
