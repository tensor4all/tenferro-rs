import json
import re
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CUDA_ARCHIVE_TEST_FILTER = (
    "-E 'not (test(eager_backend_capability_boundary) | "
    "test(execution_session_capability_cannot_project_or_escape_owner_borrow) | "
    "test(cuda_runtime_copy_into_1522_a100_destination_reuse_benchmark))'"
)


def read(path: str) -> str:
    text = (ROOT / path).read_text()
    if path == ".github/workflows/runpod-gpu-test.yml":
        # Existing execution contracts span the parent and its trusted callee.
        execution = (ROOT / ".github/workflows/runpod-gpu-execute.yml").read_text()
        index = text.index("  ci-gpu-gate:")
        text = text[:index] + execution[execution.index("  start-runpod:"):] + text[index:]
    return text


class WorkflowContractTests(unittest.TestCase):
    def test_paid_lifecycle_alone_holds_global_queue(self) -> None:
        parent = (ROOT / ".github/workflows/runpod-gpu-test.yml").read_text()
        child = (ROOT / ".github/workflows/runpod-gpu-execute.yml").read_text()
        self.assertEqual(
            parent.split("\nenv:\n", 1)[1].split("\njobs:\n", 1)[0],
            child.split("\nenv:\n", 1)[1].split("\njobs:\n", 1)[0],
        )
        self.assertNotIn("\nconcurrency:", parent)
        self.assertIn("group: runpod-tenferro-gpu-refs/heads/main", child)
        self.assertIn("cancel-in-progress: false\n  queue: max", child)
        call = parent.split("  gpu-execution:", 1)[1].split("  ci-gpu-gate:", 1)[0]
        self.assertIn("needs: [authorize, runpod-contract, pre-runpod-gate, cuda-archive]", call)
        self.assertIn("uses: ./.github/workflows/runpod-gpu-execute.yml", call)
        self.assertIn("GPU_EXECUTION_RESULT: ${{ needs.gpu-execution.result }}", parent)
        self.assertIn('record_result "gpu-execution (including cleanup)"', parent)
        self.assertNotIn("workflow_dispatch:", child)
        self.assertIn("${GITHUB_REPOSITORY}/.github/workflows/runpod-gpu-test.yml@refs/heads/main", child)
        self.assertLess(child.index("Revalidate queued PR"), child.index("Create GitHub App token"))
        self.assertIn("    if: always()", child.split("  cleanup-runpod:", 1)[1])
        self.assertIn('::error::Failed to delete RunPod pod', child)
        self.assertNotIn("actions/cache/save", child)
        gpu = child.split("  run-gpu-tests:", 1)[1].split("  cleanup-runpod:", 1)[0]
        self.assertNotIn("secrets.", gpu)

    def test_queued_head_guard_rejects_stale_closed_and_forked_prs(self) -> None:
        child = (ROOT / ".github/workflows/runpod-gpu-execute.yml").read_text()
        step = child.split("      - name: Revalidate queued PR before provisioning\n", 1)[1].split("      - name:", 1)[0]
        script = textwrap.dedent(step.split("        run: |\n", 1)[1])
        for head, base, state, repo, expected in (
            ("head", "base", "open", "tensor4all/tenferro-rs", 0),
            ("new", "base", "open", "tensor4all/tenferro-rs", 1),
            ("head", "new", "open", "tensor4all/tenferro-rs", 1),
            ("head", "base", "closed", "tensor4all/tenferro-rs", 1),
            ("head", "base", "open", "fork/tenferro-rs", 1),
        ):
            with self.subTest(head=head, base=base, state=state, repo=repo), tempfile.TemporaryDirectory() as directory:
                env = dict(os.environ, GITHUB_REPOSITORY="tensor4all/tenferro-rs", PR_NUMBER="1", EXPECTED_HEAD="head", EXPECTED_BASE="base")
                env["PR_JSON"] = json.dumps({"state": state, "head": {"sha": head, "repo": {"full_name": repo}}, "base": {"sha": base}})
                result = subprocess.run(
                    ["bash", "-c", 'gh() { printf "%s\\n" "$PR_JSON"; };\n' + script.replace("/tmp/queued-pr.json", directory + "/pr.json")],
                    env=env, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_fast_ci_uses_shared_policy_and_profiles(self) -> None:
        text = read(".github/workflows/ci.yml")
        self.assertIn("python3 scripts/ci/change_policy.py", text)
        self.assertIn("python3 scripts/ci/run_profile.py fmt", text)
        self.assertIn("python3 scripts/ci/run_profile.py blas-inject", text)
        self.assertIn("python3 scripts/ci/run_profile.py coverage", text)
        self.assertIn("python3 scripts/ci/run_profile.py docs", text)
        self.assertIn("name: CI configuration checks", text)

    def test_macos_workspace_tests_run_in_parallel_with_linux(self) -> None:
        text = read(".github/workflows/ci-pr-workspace-tests.yml")
        start = text.index("  macos:")
        block = text[start:]
        self.assertIn("name: macOS workspace tests", block)
        self.assertIn("needs: changes", block)
        self.assertNotIn("needs: [changes, ci-gate]", block)
        self.assertIn("run_macos: ${{ steps.policy.outputs.run_macos }}", text)
        self.assertIn("'macos-15' || 'ubuntu-latest'", block)
        self.assertIn("python3 scripts/ci/run_profile.py macos-accelerate", block)
        self.assertIn("shared-key: macos-accelerate-v1", block)
        self.assertNotIn("run_profile.py workspace-faer", block)
        self.assertIn("needs.changes.result == 'success'", block)
        self.assertIn("Change classification failed", block)
        self.assertNotIn("needs.ci-gate", block)
        self.assertNotIn("Linux workspace gate failed", block)

        fast = read(".github/workflows/ci.yml")
        self.assertNotIn("macOS-gated GPU type-check", fast)

    def test_coverage_installs_nextest(self) -> None:
        block = read(".github/workflows/ci.yml").split("\n  coverage:\n", 1)[1].split("\n  docs-site:\n", 1)[0]
        self.assertIn("tool: cargo-llvm-cov,nextest", block)

    def test_required_names_remain_stable(self) -> None:
        fast = read(".github/workflows/ci.yml")
        heavy = read(".github/workflows/ci-pr-workspace-tests.yml")
        for name in (
            "rustfmt",
            "clippy",
            "coverage",
            "docs-site",
            "cargo test (blas inject)",
        ):
            with self.subTest(name=name):
                self.assertIn(f"name: {name}", fast)
        self.assertIn("name: CI gate (PR workspace tests)", heavy)
        self.assertIn("name: macOS workspace tests", heavy)

    def test_gpu_gates_start_after_lint_only(self) -> None:
        for path in (".github/workflows/CI_gpu.yml", ".github/workflows/runpod-gpu-test.yml"):
            with self.subTest(path=path):
                text = read(path)
                self.assertIn('const required = ["rustfmt", "clippy"];', text)
                self.assertNotIn('              "coverage",', text)
                self.assertNotIn('              "docs-site",', text)

    def test_fast_required_jobs_fail_if_policy_fails(self) -> None:
        text = read(".github/workflows/ci.yml")
        self.assertGreaterEqual(text.count("needs.policy.result"), 6)
        self.assertIn("Change classification failed", text)

    def test_oracle_replay_nightly_is_gated_by_recent_default_branch_commits(self) -> None:
        text = read(".github/workflows/oracle-replay-nightly.yml")
        self.assertIn("name: Oracle replay nightly", text)
        self.assertIn("schedule:", text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("fetch-depth: 0", text)
        self.assertIn("git log --since=\"24 hours ago\"", text)
        self.assertIn("run_oracle=true", text)
        self.assertIn("run_oracle=false", text)
        self.assertIn("github.event.inputs.force == 'true'", text)
        self.assertIn("RUN_ORACLE_REPLAY=1", text)
        self.assertIn("ORACLE_REPLAY_JOBS", text)
        self.assertIn("oracle_replays_supported_db_cases_when_requested", text)
        self.assertIn(
            "Swatinem/rust-cache@e18b497796c12c097a38f9edb9d0641fb99eee32",
            text,
        )
        self.assertIn("prefix-key: v1-rust-oracle-replay-ubuntu22", text)
        self.assertIn("shared-key: oracle-replay-autodiff", text)
        self.assertIn("cache-all-crates: true", text)
        self.assertIn("cache-workspace-crates: true", text)
        self.assertIn("workspaces: . -> target", text)
        self.assertIn("save-if: ${{ github.ref == 'refs/heads/main' }}", text)
        self.assertIn("actions/upload-artifact@", text)
        self.assertIn("Oracle replay not required", text)

    def test_heavy_workflow_has_explicit_noop_matrix_and_gate_contract(self) -> None:
        text = read(".github/workflows/ci-pr-workspace-tests.yml")
        self.assertIn('"backend":"not-required"', text)
        self.assertIn("RUN_WORKSPACE", text)
        self.assertIn("RUN_EXTENSIONS", text)
        self.assertIn("Workspace tests not required", text)
        self.assertIn("python3 scripts/ci/run_profile.py", text)
        self.assertNotIn("grep -qE", text)
        self.assertIn(". -> ../target-${{ matrix.cfg.backend }}", text)
        self.assertIn('libfaer-*.rlib', text)
        self.assertIn("strided_view strided_traits strided_perm", text)

    def test_ci_config_installs_a_pinned_actionlint(self) -> None:
        text = read(".github/workflows/ci.yml")
        self.assertIn(
            "github.com/rhysd/actionlint/cmd/actionlint@v1.7.7", text
        )

    def test_changed_error_audit_jobs_fetch_base_history(self) -> None:
        job_header = re.compile(r"^  (?P<name>[A-Za-z0-9_-]+):\s*$", re.MULTILINE)
        audited_jobs: list[str] = []
        workflow_dir = ROOT / ".github" / "workflows"
        for path in sorted(workflow_dir.iterdir()):
            if path.suffix not in {".yml", ".yaml"}:
                continue
            text = path.read_text()
            matches = list(job_header.finditer(text))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                block = text[match.start() : end]
                if "check-public-error-docs.py --changed-from" not in block:
                    continue
                job = f"{path.name}:{match.group('name')}"
                audited_jobs.append(job)
                self.assertIn("uses: actions/checkout@", block, job)
                self.assertRegex(block, r"(?m)^\s+fetch-depth:\s*0\s*$", job)

        self.assertTrue(audited_jobs, "no changed public-error audit job was found")

    def test_runpod_schema_preflight_precedes_archive(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        self.assertIn("runpod-contract:", text)
        self.assertIn("python3 scripts/ci/runpod_contract.py", text)
        archive = text.index("  cuda-archive:")
        preflight = text.index("  runpod-contract:")
        self.assertLess(preflight, archive)
        archive_block = text[archive : text.index("  start-runpod:")]
        self.assertIn("- runpod-contract", archive_block)

    def test_runpod_gpu_skip_uses_trusted_authorize_output(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        self.assertIn("gpu_required: ${{ steps.resolve_ref.outputs.run_gpu }}", text)
        self.assertIn("gh api --paginate", text)
        self.assertIn("python3 scripts/ci/change_policy.py", text)
        self.assertIn("GPU validation not required", text)
        self.assertIn("GPU_REQUIRED: ${{ needs.authorize.outputs.gpu_required }}", text)

    def test_runpod_authorization_uses_repository_roles(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        authorization = text[
            text.index("          check_permission() {") : text.index(
                "          classify_pr_paths() {"
            )
        ]
        self.assertIn("--jq .role_name", authorization)
        self.assertNotIn("--jq .permission", authorization)
        self.assertIn("admin|maintain)", authorization)
        self.assertEqual(
            re.findall(r'^[ \t]+check_permission "[^\n]+$', text, re.MULTILINE),
            [
                '            check_permission "${pr_author}" "PR author"',
                '            check_permission "${WORKFLOW_RUN_ACTOR}" "Source workflow actor"',
                '            check_permission "${GITHUB_ACTOR}" "Workflow actor"',
                '              check_permission "${pr_author}" "PR author"',
            ],
        )

    def test_review_labels_are_authorized_by_repository_role(self) -> None:
        text = read(".github/workflows/review_bot.yml")
        self.assertEqual(
            text.count('const allowed = new Set(["admin", "maintain"]);'), 2
        )
        self.assertEqual(text.count("allowed.has(data.role_name)"), 2)
        self.assertNotIn("allowed.has(data.permission)", text)
        self.assertEqual(text.count("has ${data.role_name} repository role"), 2)

    def test_runpod_secret_stays_on_trusted_hosted_jobs(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        run_gpu = text[
            text.index("  run-gpu-tests:") : text.index("  cleanup-runpod:")
        ]
        self.assertNotIn("RUNPOD_API_KEY", run_gpu)
        self.assertIn("RUNPOD_API_KEY", text[text.index("  runpod-contract:") :])

    def test_runpod_creation_uses_status_aware_helper(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        create = text[
            text.index(
                "      - name: Provision cheapest compatible RunPod pod"
            ) : text.index("      - name: Delete pod if runner startup failed")
        ]
        self.assertIn("python3 -m scripts.ci.runpod_provision", create)
        self.assertNotIn("python3 scripts/ci/runpod_client.py", create)
        self.assertNotIn("for attempt in $(seq 1 5)", create)
        self.assertNotIn("curl -sS", create)

    def test_runpod_smoke_proof_gates_runner_registration(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        create = text[
            text.index(
                "      - name: Provision cheapest compatible RunPod pod"
            ) : text.index("      - name: Delete pod if runner startup failed")
        ]
        # The smoke proof must run inside the startup script BEFORE the
        # runner registers, and its script must be fetched at the trusted
        # default-branch SHA — never from a PR-controlled ref.
        smoke = create.index("cuda_smoke_test.py")
        runner = create.index("./run.sh --jitconfig")
        self.assertLess(smoke, runner)
        # The smoke script is embedded from the trusted checkout, never
        # fetched over the network from the pod (raw.githubusercontent is
        # rate-limited from datacenter IPs and its failure looked like a
        # startup timeout in live runs).
        self.assertIn("cat scripts/ci/cuda_smoke_test.py >> /tmp/runpod-startup.sh", create)
        self.assertNotIn("raw.githubusercontent.com", create)
        embed = create.index("cat > /tmp/cuda_smoke_test.py <<'EMBEDDED_SMOKE_PY'")
        self.assertLess(embed, create.index("env -u RUNNER_JIT_CONFIG python3 /tmp/cuda_smoke_test.py"))
        # The single credential that reaches the pod (the one-shot JIT
        # runner config) must be stripped from the smoke child's env.
        self.assertIn("env -u RUNNER_JIT_CONFIG python3 /tmp/cuda_smoke_test.py", create)
        # Debug switch: keep smoke-rejected pods for console-log triage.
        self.assertIn("PROVISION_KEEP_FAILED_PODS: ${{ inputs.keep_failed_pods || 'false' }}", create)
        # The stale fetch env must be fully gone: a leftover expansion under
        # set -u would abort every provision run before pod creation.
        whole = read(".github/workflows/runpod-gpu-test.yml")
        self.assertNotIn("SMOKE_SOURCE_URL", whole)
        # Debug retention must also gate the workflow-side deletion paths,
        # or the cleanup steps would delete the pod being inspected.
        self.assertIn(
            "if: failure() && steps.create_pod.outputs.pod_id != '' && inputs.keep_failed_pods != true",
            whole,
        )
        self.assertIn(
            "if: inputs.keep_failed_pods != true || needs.start-runpod.result == 'success'",
            whole,
        )
        # JIT configs are minted per candidate attempt inside the provision
        # loop; the workflow must not pre-mint a single shared config, and
        # run-gpu-tests must target the ACCEPTED attempt's label.
        text_full = read(".github/workflows/runpod-gpu-test.yml")
        self.assertNotIn("- name: Generate JIT runner config", text_full)
        self.assertNotIn("RUNNER_JIT_CONFIG: ${{", text_full)
        self.assertIn(
            "runner_label: ${{ steps.create_pod.outputs.runner_label }}",
            text_full,
        )
        self.assertIn("PROVISION_RUNNER_GROUP_ID:", create)
        # The build toolchain, git, jq, and zstd are installed by the test
        # job's own first step: registration and the smoke proof must not wait
        # for them, because every pre-registration second is billed at the GPU
        # rate and a rejected candidate pays it too. zstd still lands before
        # the actions/cache restore step, which is what keeps the cache version
        # hash compatible with the zstd-equipped hosted publisher.
        self.assertIn("RUNNER_CACHE_DIR=\"/workspace/runpod-ci-cache\"", create)
        self.assertIn('echo "${RUNNER_SHA256}  ${RUNNER_TARBALL}" | sha256sum -c', create)
        self.assertLess(
            create.index('echo "${RUNNER_SHA256}  ${RUNNER_CACHED_TARBALL}"'),
            create.index("curl -fsSL -o \"${RUNNER_TARBALL}\""),
            "a cached runner tarball must be considered before downloading",
        )
        # A missing, unwritable, or stale cache must fall back to the download
        # instead of failing the startup script under `set -e`.
        self.assertIn("2>/dev/null || true", create)
        self.assertIn('echo "warning: could not populate the runner tarball cache"', create)
        whole_job = read(".github/workflows/runpod-gpu-test.yml")
        install = whole_job.index("      - name: Install pod-side build dependencies")
        job_install = whole_job[
            install : whole_job.index("      - name: Checkout tenferro-rs", install)
        ]
        for package in ("zstd \\", "git \\", "jq \\", "build-essential \\"):
            self.assertIn(package, job_install)
            self.assertNotIn(package, create)
        self.assertLess(
            install,
            whole_job.index("      - name: Restore CUDA/PJRT test archives"),
            "zstd must land before the cache restore step",
        )
        # The smoke's NVRTC-only install leaves a partial /usr/local tree;
        # the test job's runtime discovery must reject trees missing the
        # full library set instead of skipping the real runtime install.
        text = read(".github/workflows/runpod-gpu-test.yml")
        configure = text[
            text.index("      - name: Configure CUDA runtime libraries") : text.index(
                "      - name: Verify loaded NVRTC version"
            )
        ]
        self.assertIn("cuda_tree_has_runtime_libs", configure)
        for lib in ("libcublas.so", "libcusolver.so", "libcusparse.so", "libnvrtc.so"):
            self.assertIn(lib, configure)
        # Both acceptance paths (discovered toolkit and cached seed tree)
        # must run the completeness check.
        self.assertGreaterEqual(configure.count("cuda_tree_has_runtime_libs "), 2)
        self.assertNotIn("TENFERRO_REF", create)
        # Smoke parameters flow through non-secret pod env only.
        for pod_env in (
            "SMOKE_MIN_RUNTIME_VERSION=",
            "SMOKE_FULL_RUNTIME_VERSION=",
            "SMOKE_MIN_VRAM_GB=",
        ):
            self.assertIn(f'--pod-env "{pod_env}', create)
        self.assertNotIn('--pod-env "RUNPOD_API_KEY', create)

    def test_the_paid_workflow_applies_the_label_decision_in_a_step(self) -> None:
        """Job-level `if`s in the called workflow did not see the inputs.

        A live labelled dispatch showed `Revalidate queued PR before
        provisioning` running (so `inputs` reaches steps) while `start-runpod`
        was still scheduled from its job `if`. The decision is therefore applied
        inside a step that also skips provisioning.
        """

        execute = read(".github/workflows/runpod-gpu-execute.yml")
        self.assertIn("id: local_gpu_validation", execute)
        self.assertIn("paid_path_skipped: ${{ steps.local_gpu_validation.outputs.skip }}", execute)
        # The label read is the same text the parent authorizes on.
        self.assertIn('grep -Fxq "gpu-validated-locally"', execute)
        # Provisioning and the runner wait both depend on the step output.
        provision = execute[
            execute.index("      - name: Provision cheapest compatible RunPod pod") : execute.index(
                "      - name: Provision cheapest compatible RunPod pod"
            )
            + 300
        ]
        self.assertIn("if: steps.local_gpu_validation.outputs.skip != 'true'", provision)
        tests_job = execute[
            execute.index("  run-gpu-tests:") : execute.index("  run-gpu-tests:") + 400
        ]
        self.assertIn("needs.start-runpod.outputs.paid_path_skipped != 'true'", tests_job)

    def test_the_paid_workflow_decides_from_its_own_inputs(self) -> None:
        """The decision must travel as inputs, not as a caller `if`.

        Measured behaviour: a skipped caller job does not pass its `with:` block
        and the called workflow's jobs are scheduled anyway, so a labelled PR
        began provisioning pods. Every paid job therefore applies the inputs
        itself, and the caller only forwards them.
        """

        execute = read(".github/workflows/runpod-gpu-execute.yml")
        for input_name in ("gpu_required:", "local_gpu_validation:"):
            self.assertIn(input_name, execute)
        expected = "inputs.gpu_required == 'true' &&\n      inputs.local_gpu_validation != 'true'"
        for job in ("  start-runpod:", "  run-gpu-tests:"):
            block = execute[execute.index(job) : execute.index(job) + 500]
            self.assertIn(expected, block, job)
        text = read(".github/workflows/runpod-gpu-test.yml")
        self.assertIn("gpu_required: ${{ needs.authorize.outputs.gpu_required }}", text)
        self.assertIn(
            "local_gpu_validation: ${{ needs.authorize.outputs.local_gpu_validation }}",
            text,
        )
        # The caller must not gate itself: that is what dropped the inputs.
        # The spliced view inserts the callee's jobs next, so bound the slice
        # at the first callee job.
        caller = text[
            text.index("  gpu-execution:") : text.index("  start-runpod:", text.index("  gpu-execution:"))
        ]
        self.assertNotIn("if:", caller, "the caller job must not gate the call")
        # Evidence is verified whenever the label is present.
        self.assertIn("if [ \"${LOCAL_GPU_VALIDATION}\" = true ]; then", text)

    def test_merged_or_closed_pulls_do_not_provision_pods(self) -> None:
        """A gate run that completes after the merge must not pay for a pod."""

        text = read(".github/workflows/runpod-gpu-test.yml")
        self.assertIn('pr_state="$(jq -r \'.state\' <<<"${pr_json}")"', text)
        self.assertIn('if [ "${pr_state}" != "open" ]; then', text)
        self.assertIn("reason=pull request is ${pr_state}", text)
        # The state check has to come before the pull merge ref, which stops
        # existing once the PR is merged or closed.
        self.assertLess(
            text.index('pr_state="$(jq -r'), text.index("git/ref/pull/${pr_number}/merge")
        )

    def test_local_gpu_validation_label_substitutes_for_the_paid_gate(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        # The label decision comes from the PR's labels and skips the whole
        # paid path, so an outage cannot be paid for repeatedly.
        self.assertIn('grep -Fxq "gpu-validated-locally"', text)
        self.assertIn("local_gpu_validation: ${{ steps.resolve_ref.outputs.local_gpu_validation }}", text)
        # The decision reaches the paid workflow as an input, which its own jobs
        # apply (see test_the_paid_workflow_decides_from_its_own_inputs).
        self.assertIn(
            "local_gpu_validation: ${{ needs.authorize.outputs.local_gpu_validation }}",
            text,
        )
        # The label alone must not waive the gate: the evidence comment and its
        # author's repository role are both verified before success is published.
        self.assertIn("Local GPU validation:", text)
        self.assertIn("verify_local_gpu_evidence", text)
        self.assertIn(
            'collaborators/${evidence_login}/permission', text
        )
        # Passing requires an accepted evidence note, not just the label.
        self.assertIn("local_gpu_note", text)
        self.assertIn("RunPod CI GPU gate passed (local GPU validation)", text)
        # The rule and the evidence format are documented where contributors and
        # reviewers look for them.
        for path in ("REPOSITORY_RULES.md", "CONTRIBUTING.md", "docs/design/runpod-gpu-provisioning.md"):
            doc = read(path)
            self.assertIn("gpu-validated-locally", doc, path)
            self.assertIn("Local GPU validation:", doc, path)

    def test_runpod_provision_is_bounded_and_price_ordered(self) -> None:
        config = json.loads(read("scripts/ci/runpod_config.json"))
        for key in (
            "graphql_url",
            "min_vram_gb",
            "max_price_candidates",
            "max_provision_attempts",
            "max_consecutive_startup_failures",
            "startup_timeout_seconds",
            "startup_poll_seconds",
        ):
            self.assertIn(key, config)
        self.assertGreaterEqual(config["max_provision_attempts"], 1)
        self.assertLessEqual(config["max_provision_attempts"], 8)
        # Live-priced candidates must never starve the reviewed static
        # fallback tiers out of the bounded attempt budget.
        self.assertGreaterEqual(
            config["max_provision_attempts"],
            config["max_price_candidates"] + len(config["gpu_tiers"]),
        )
        text = read(".github/workflows/runpod-gpu-test.yml")
        self.assertIn("gpu_cost_per_hr:", text)
        self.assertIn("RunPod hourly price:", text)
        self.assertIn("RunPod estimated paid cost:", text)
        # The job timeout must contain the worst-case provision budget so
        # the loop reaches its explicit exhaustion error instead of being
        # cancelled mid-attempt (60s deletion + 300s setup margins).
        start_runpod = text[
            text.index("  start-runpod:") : text.index("  run-gpu-tests:")
        ]
        timeout = re.search(r"timeout-minutes: (\d+)", start_runpod)
        assert timeout is not None
        worst_case = config["max_provision_attempts"] * (
            config["create_deadline_seconds"]
            + config["startup_timeout_seconds"]
            + 60
        )
        self.assertGreaterEqual(int(timeout.group(1)) * 60, worst_case + 300)

    def test_runpod_selected_gpu_is_forwarded_and_logged(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        self.assertIn(
            "gpu_tier: ${{ steps.create_pod.outputs.gpu_tier }}", text
        )
        self.assertIn("needs.start-runpod.outputs.gpu_type_id", text)
        self.assertIn("needs.start-runpod.outputs.gpu_tier", text)
        self.assertIn("nvidia-smi --query-gpu=index,name", text)
        check_machine = text[
            text.index("      - name: Check machine") : text.index(
                "      - name: Restore cuTENSOR redistributable"
            )
        ]
        run_script = check_machine[check_machine.index("        run: |") :]
        self.assertNotIn("${{ needs.start-runpod.outputs", run_script)
        self.assertIn("${RUNPOD_GPU_TYPE_ID}", run_script)
        self.assertIn("${RUNPOD_GPU_TIER}", run_script)

    def test_runpod_rejected_gpu_still_reaches_cleanup(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        client = read("scripts/ci/runpod_client.py")
        main = client[client.index("def main()") :]
        self.assertIn("except AssignedGpuError as error:", main)
        self.assertIn("publish_cleanup_pod_id(", main)
        startup_cleanup = text[
            text.index("      - name: Delete pod if runner startup failed") :
            text.index("  run-gpu-tests:")
        ]
        self.assertIn("steps.create_pod.outputs.pod_id != ''", startup_cleanup)
        self.assertIn("POD_ID: ${{ steps.create_pod.outputs.pod_id }}", startup_cleanup)

    def test_runpod_cache_key_uses_content_not_ref_identity(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        key_line = next(
            line
            for line in text.splitlines()
            if 'key="cuda-pjrt-archive-' in line or 'key="cuda-archive-' in line
        )
        self.assertNotIn("TENFERRO_REF", key_line)
        self.assertIn("hashFiles(", key_line)
        # Material build inputs only (#1403): workflow YAML edits must not
        # invalidate the archive key, but everything executed from the
        # checkout during the build (scripts/ci/**, .cargo/**) MUST be
        # hashed so an artifact name match proves those inputs were
        # identical too.
        self.assertNotIn(".github/workflows", key_line)
        self.assertIn("rust${rustc_version}", key_line)
        for material in (
            "tenferro-rs/Cargo.lock",
            "tenferro-rs/**/Cargo.toml",
            "tenferro-rs/**/src/**",
            "tenferro-rs/**/tests/**",
            "tenferro-rs/**/examples/**",
            "tenferro-rs/**/benches/**",
            "tenferro-rs/**/build.rs",
            "tenferro-rs/scripts/ci/**",
            "tenferro-rs/.cargo/**",
            "tenferro-rs/rust-toolchain*",
        ):
            self.assertIn(material, key_line)

    def test_cuda_archives_use_cargo_ci_profile_not_release(self) -> None:
        for path in (
            ".github/workflows/runpod-gpu-test.yml",
            ".github/workflows/CI_gpu.yml",
        ):
            text = read(path)
            with self.subTest(path=path):
                self.assertIn("cargo nextest archive", text)
                self.assertEqual(text.count("--cargo-profile ci"), text.count("cargo nextest archive"))
                self.assertNotIn("nextest archive \\\n            --release", text)
                self.assertNotIn("nextest archive \\\n              --release", text)
                for match in re.finditer(r"cargo nextest archive[\s\S]{0,280}", text):
                    self.assertNotIn("--release", match.group(0))

    def test_gpu_archive_run_excludes_compile_only_trybuild_tests(self) -> None:
        # RunPod's exhaustive inventory is checked by test_gpu_test_partition.
        for path in (".github/workflows/CI_gpu.yml",):
            text = read(path)
            cuda_tests = text[
                text.index("      - name: Run CUDA tests from archive") :
                text.index("      - name: Run OpenXLA PJRT E2E tests from archive")
            ]
            filter_lines = [
                line.strip()
                for line in cuda_tests.splitlines()
                if line.strip().startswith("-E ")
            ]
            with self.subTest(path=path):
                self.assertEqual(filter_lines, [f"{CUDA_ARCHIVE_TEST_FILTER} \\"])

        for path in (
            "crates/tenferro-ad/tests/integration/eager_backend_capability_contract.rs",
            "crates/tenferro-gpu/tests/integration/session_contract.rs",
        ):
            source = read(path)
            with self.subTest(nextest_archive_guard=path):
                self.assertIn('var_os("NEXTEST")', source)
                self.assertNotIn('var("CARGO_NET_OFFLINE")', source)

    def test_cuda_correctness_gate_excludes_a100_performance_benchmark(self) -> None:
        benchmark = "cuda_runtime_copy_into_1522_a100_destination_reuse_benchmark"
        structural_tests = read(
            "crates/tenferro-gpu/src/cubecl/tests/structural_tests.rs"
        )
        self.assertRegex(
            structural_tests,
            rf"#\[ignore[^\]]*\]\s*fn {benchmark}\(\)",
        )
        for path in (".github/workflows/CI_gpu.yml",):
            text = read(path)
            cuda_tests = text[
                text.index("      - name: Run CUDA tests from archive") :
                text.index("      - name: Run OpenXLA PJRT E2E tests from archive")
            ]
            filter_lines = [
                line.strip()
                for line in cuda_tests.splitlines()
                if line.strip().startswith("-E ")
            ]
            with self.subTest(path=path):
                self.assertEqual(filter_lines, [f"{CUDA_ARCHIVE_TEST_FILTER} \\"])

    def test_pjrt_uses_hosted_archive_not_runpod_cargo(self) -> None:
        for path in (
            ".github/workflows/runpod-gpu-test.yml",
            ".github/workflows/CI_gpu.yml",
        ):
            text = read(path)
            with self.subTest(path=path):
                self.assertIn("PJRT_ARCHIVE:", text)
                self.assertIn("pjrt-tests.tar.zst", text)
                self.assertIn("-p tenferro-xla", text)
                self.assertIn("--features pjrt", text)
                self.assertIn("Build PJRT test archive", text)
                self.assertIn("Run OpenXLA PJRT E2E tests from archive", text)
                self.assertIn("--archive-file \"${PJRT_ARCHIVE}\"", text)
                if path.endswith("CI_gpu.yml"):
                    self.assertIn("-E 'test(pjrt_execution)'", text)
                else:
                    self.assertIn("gpu_test_partition.py --kind pjrt --lane gpu", text)
                self.assertNotIn("cargo test -p tenferro-xla", text)

    def test_runpod_cuda_runtime_adapts_without_lowering_cudarc_bindings(
        self,
    ) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        cargo = read("Cargo.toml")
        runpod_config = json.loads(read("scripts/ci/runpod_config.json"))
        workflow_cudarc = re.search(
            r'^  CUDARC_CUDA_VERSION: "(\d+)"$', text, re.MULTILINE
        )
        runtime = re.search(
            r'^  CUDA_RUNTIME_VERSION: "(\d+)\.(\d+)"$',
            text,
            re.MULTILINE,
        )
        minimum_runtime = re.search(
            r'^  CUDA_MIN_RUNTIME_VERSION: "(\d+)\.(\d+)"$',
            text,
            re.MULTILINE,
        )
        cargo_cudarc = re.search(
            r'^cudarc = \{[^\n]*features = \[[^\n]*"cuda-(\d+)"',
            cargo,
            re.MULTILINE,
        )
        self.assertIsNotNone(workflow_cudarc)
        self.assertIsNotNone(runtime)
        self.assertIsNotNone(minimum_runtime)
        self.assertIsNotNone(cargo_cudarc)
        assert workflow_cudarc is not None
        assert runtime is not None
        assert minimum_runtime is not None
        assert cargo_cudarc is not None
        encoded_runtime = (
            int(runtime.group(1)) * 1000 + int(runtime.group(2)) * 10
        )
        self.assertEqual(int(workflow_cudarc.group(1)), encoded_runtime)
        self.assertEqual(workflow_cudarc.group(1), cargo_cudarc.group(1))
        full_runtime = (int(runtime.group(1)), int(runtime.group(2)))
        minimum = (
            int(minimum_runtime.group(1)),
            int(minimum_runtime.group(2)),
        )
        self.assertEqual(minimum, (12, 6))
        allowed_versions = [
            tuple(int(part) for part in version.split("."))
            for version in runpod_config["allowed_cuda_versions"]
        ]
        self.assertIn(minimum, allowed_versions)
        self.assertIn(full_runtime, allowed_versions)
        self.assertTrue(
            all(version >= minimum for version in allowed_versions)
        )
        self.assertIn("id: select_cuda_runtime", text)
        self.assertIn("Selected CUDA runtime:", text)
        self.assertIn("steps.select_cuda_runtime.outputs.runtime_version", text)
        self.assertIn("nvrtc.nvrtcVersion", text)
        self.assertIn("Loaded NVRTC version:", text)
        self.assertIn("if loaded < minimum:", text)
        self.assertIn("if loaded > driver:", text)
        self.assertIn("newer than driver", text)

    def test_manual_pr_recovery_is_authorized_and_head_stable(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        self.assertIn("pr_number:", text)
        self.assertIn("MANUAL_PR_NUMBER", text)
        self.assertIn(".state", text)
        self.assertIn(".head.repo.full_name", text)
        self.assertIn("refusing recovery", text)
        self.assertIn(
            "target_head_sha: ${{ steps.resolve_ref.outputs.target_head_sha }}",
            text,
        )
        gate = text[text.index("  ci-gpu-gate:") :]
        self.assertIn(
            "TARGET_HEAD_SHA: ${{ needs.authorize.outputs.target_head_sha }}",
            gate,
        )
        self.assertNotIn("WORKFLOW_RUN_PULL_REQUESTS", gate)

    def test_runpod_gate_only_runs_for_eligible_trigger(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        gate = text[text.index("  ci-gpu-gate:") :]
        condition = re.search(r"    if: >-\n(.*?)\n\n", gate, re.DOTALL)
        self.assertIsNotNone(condition)
        assert condition is not None
        expression = " ".join(condition.group(1).split())
        # Evaluate the actual workflow condition across upstream outcomes.
        # Failed authorization must still reach the aggregate failure check.
        for event in ("workflow_dispatch", "workflow_run"):
            for source in ("pull_request", "push"):
                for conclusion in ("success", "failure", "cancelled", "skipped"):
                    for status in ("in_progress", "completed"):
                        for cancelled in (False, True):
                            with self.subTest(
                                event=event, source=source,
                                conclusion=conclusion, status=status,
                                cancelled=cancelled,
                            ):
                                translated = expression
                                for key, value in {
                                    "github.event_name": event,
                                    "github.event.workflow_run.event": source,
                                    "github.event.workflow_run.conclusion": conclusion,
                                    "github.event.workflow_run.status": status,
                                    "always()": True,
                                    "cancelled()": cancelled,
                                }.items():
                                    translated = translated.replace(key, repr(value))
                                translated = translated.replace("&&", " and ")
                                translated = translated.replace("||", " or ")
                                translated = translated.replace("!", " not ")
                                actual = eval(translated, {"__builtins__": {}})
                                expected = not cancelled and (
                                    event == "workflow_dispatch"
                                    or (source == "pull_request" and status == "in_progress")
                                )
                                self.assertEqual(actual, expected)

    def test_runpod_workflow_is_cache_reader_only(self) -> None:
        """No job that builds PR code or runs on the pod may write caches (#1403)."""

        text = read(".github/workflows/runpod-gpu-test.yml")
        self.assertIn("types: [in_progress]", text)
        self.assertNotIn("types: [completed]", text)
        self.assertNotIn("uses: actions/cache@", text)
        self.assertNotIn("actions/cache/save", text)
        rust_cache_uses = text.count("Swatinem/rust-cache")
        self.assertEqual(
            len(re.findall(r"(?m)^\s+save-if: false$", text)), rust_cache_uses
        )
        top = text[: text.index("jobs:")]
        permissions = top[top.index("permissions:") :]
        permissions = permissions[: permissions.index("\n\n")]
        self.assertNotIn("write", permissions)
        # The only extra job scope is actions: read for artifact lookup.
        for match in re.finditer(r"(?m)^      actions: (\S+)$", text):
            self.assertEqual(match.group(1), "read")

    def test_cache_publish_is_default_branch_only(self) -> None:
        text = read(".github/workflows/ci-cache-publish.yml")
        triggers = text[text.index("on:") : text.index("concurrency:")]
        self.assertIn("push:", triggers)
        self.assertIn("branches: [main]", triggers)
        self.assertNotIn("pull_request", triggers)
        self.assertNotIn("workflow_run", triggers)
        # The writer must only build code from the trusted default branch:
        # its checkouts must not override the ref, and every job must refuse
        # to run when a workflow_dispatch selected a non-main ref.
        self.assertNotIn("ref:", text)
        self.assertIn("actions: write", text)
        job_count = len(re.findall(r"(?m)^  [a-z][a-z0-9-]*:$", text[text.index("jobs:") :]))
        self.assertEqual(
            text.count("if: github.ref == 'refs/heads/main'"), job_count
        )

    def test_archive_key_and_cache_ids_match_publisher_and_consumer(self) -> None:
        consumer = read(".github/workflows/runpod-gpu-test.yml")
        publisher = read(".github/workflows/ci-cache-publish.yml")

        def key_line(text: str) -> str:
            return next(
                line.strip()
                for line in text.splitlines()
                if 'key="cuda-pjrt-archive-' in line
            )

        self.assertEqual(key_line(consumer), key_line(publisher))
        for pair_line in (
            "prefix-key: v8-rust-cuda-pjrt-ci-ubuntu22-ptx",
            "shared-key: cuda-pjrt-ci-${{ env.CUDARC_CUDA_VERSION }}-ptx-${{ env.CUDA_RUNTIME_VERSION }}",
            "key: cutensor-${{ runner.os }}-x86_64-${{ env.CUTENSOR_VERSION }}-cuda12-v2",
        ):
            self.assertIn(pair_line, consumer)
            self.assertIn(pair_line, publisher)
        self.assertIn(
            "key: cuda-runtime-${{ runner.os }}-x86_64-${{ steps.select_cuda_runtime.outputs.runtime_version }}-minimal-v6",
            consumer,
        )
        self.assertIn(
            "key: cuda-runtime-${{ runner.os }}-x86_64-${{ matrix.cuda }}-minimal-v6",
            publisher,
        )
        for env_line in (
            '  CUDARC_CUDA_VERSION: "12080"',
            '  CUDA_MIN_RUNTIME_VERSION: "12.6"',
            '  CUDA_RUNTIME_VERSION: "12.8"',
            '  CUTENSOR_VERSION: "2.6.0.4"',
            "  TENFERRO_CI_CACHE_ROOT: /opt/tenferro-ci",
        ):
            self.assertIn(env_line, consumer)
            self.assertIn(env_line, publisher)

    def test_toolkit_cache_is_exact_and_main_written_only(self) -> None:
        parent = (ROOT / ".github/workflows/runpod-gpu-test.yml").read_text()
        publisher = read(".github/workflows/ci-cache-publish.yml")
        key = next(line.strip() for line in parent.splitlines() if "key: cuda-toolkit-" in line)
        self.assertIn(key, publisher)
        self.assertEqual(publisher.count(key), 2)
        self.assertIn("hashFiles('tenferro-rs/scripts/ci/install_cuda_toolkit_hosted.sh')", key)
        restore = parent.split("      - name: Restore trusted CUDA toolkit", 1)[1].split("      - name:", 1)[0]
        self.assertNotIn("restore-keys:", restore)
        self.assertNotIn("actions/cache/save", parent)
        helper = read("scripts/ci/install_cuda_toolkit_hosted.sh")
        self.assertIn('"${nvcc_bin}" --ptx -arch=sm_75', helper)
        self.assertIn("entry tenferro_toolkit_smoke", helper)

    def test_gpu_retry_reuses_immutable_artifact(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        archive_block = text[
            text.index("  cuda-archive:") : text.index("  start-runpod:")
        ]
        reuse = archive_block.index("find_archive_artifact.py")
        build = archive_block.index("Build CUDA test archive")
        self.assertLess(reuse, build)
        # Every build-path step is skipped on restore/reuse; nextest installation
        # is not a build step and is required to execute the restored archives.
        self.assertEqual(
            archive_block.count(
                "if: steps.cuda_archive_cache.outputs.cache-hit != 'true' && steps.archive_reuse.outputs.reused != 'true'"
            ),
            7,
        )
        self.assertIn(
            "name: ${{ steps.archive_key.outputs.artifact_name }}", archive_block
        )
        run_gpu = text[
            text.index("  run-gpu-tests:") : text.index("  cleanup-runpod:")
        ]
        self.assertIn(
            "name: ${{ inputs.archive_artifact_name }}",
            run_gpu,
        )

    def test_archive_key_hashes_every_embedded_markdown_input(self) -> None:
        """include_str! docs are compile-time inputs to archived binaries."""

        text = read(".github/workflows/runpod-gpu-test.yml")
        key_line = next(
            line for line in text.splitlines() if 'key="cuda-pjrt-archive-' in line
        )
        embeds: set[str] = set()
        for rust_file in (ROOT / "crates").rglob("*.rs"):
            source = rust_file.read_text()
            if "include_str!" not in source and "include_bytes!" not in source:
                continue
            for quoted in re.findall(r'"([^"]+\.md)"', source):
                repo_relative = re.sub(r"^(\.\./|/)+", "", quoted)
                self.assertTrue(
                    (ROOT / repo_relative).is_file(),
                    f"{rust_file}: cannot resolve embedded path {quoted!r}",
                )
                embeds.add(repo_relative)
        self.assertTrue(embeds, "expected at least one embedded markdown input")
        for path in sorted(embeds):
            self.assertIn(
                f"tenferro-rs/{path}",
                key_line,
                f"embedded compile-time input {path} missing from archive key",
            )

    def test_finder_only_trusts_default_branch_workflow_definitions(self) -> None:
        from scripts.ci.find_archive_artifact import (
            TRUSTED_PRODUCER_EVENTS,
            TRUSTED_WORKFLOW_PATHS,
        )

        self.assertNotIn("pull_request", TRUSTED_PRODUCER_EVENTS)
        self.assertNotIn("pull_request_target", TRUSTED_PRODUCER_EVENTS)
        for path in TRUSTED_WORKFLOW_PATHS:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_actionlint_knows_the_organization_gpu_runner(self) -> None:
        config = read(".github/actionlint.yaml")
        self.assertIn("self-hosted-runner:", config)
        self.assertIn("- ubuntu-gpu", config)
        self.assertNotIn(
            "cache-workspace-crates", read(".github/workflows/CI_gpu.yml")
        )


if __name__ == "__main__":
    unittest.main()
