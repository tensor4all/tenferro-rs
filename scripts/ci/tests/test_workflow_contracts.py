import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CUDA_ARCHIVE_TEST_FILTER = (
    "-E 'not (test(eager_backend_capability_boundary) | "
    "test(execution_session_capability_cannot_project_or_escape_owner_borrow) | "
    "test(cuda_runtime_copy_into_1522_a100_destination_reuse_benchmark))'"
)


def read(path: str) -> str:
    return (ROOT / path).read_text()


class WorkflowContractTests(unittest.TestCase):
    def test_fast_ci_uses_shared_policy_and_profiles(self) -> None:
        text = read(".github/workflows/ci.yml")
        self.assertIn("python3 scripts/ci/change_policy.py", text)
        self.assertIn("python3 scripts/ci/run_profile.py fmt", text)
        self.assertIn("python3 scripts/ci/run_profile.py blas-inject", text)
        self.assertIn("python3 scripts/ci/run_profile.py coverage", text)
        self.assertIn("python3 scripts/ci/run_profile.py docs", text)
        self.assertIn("name: CI configuration checks", text)

    def test_macos_gated_gpu_tests_are_cross_checked(self) -> None:
        text = read(".github/workflows/ci.yml")
        start = text.index("  macos-gated-check:")
        end = text.index("\n  coverage:", start)
        block = text[start:end]
        self.assertIn("name: macOS-gated GPU type-check", block)
        self.assertIn("targets: aarch64-apple-darwin", block)
        self.assertIn(
            "cargo check -p tenferro-gpu --features webgpu --test integration "
            "--target aarch64-apple-darwin",
            block,
        )
        self.assertIn("needs.policy.outputs.run_rust == 'true'", block)

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
        # zstd on the pod keeps the actions/cache version hash compatible
        # with the zstd-equipped hosted publisher; without it every pod
        # restore misses exact-match keys.
        self.assertIn("zstd \\", create)
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

    def test_runpod_provision_is_bounded_and_price_ordered(self) -> None:
        config = json.loads(read("scripts/ci/runpod_config.json"))
        for key in (
            "graphql_url",
            "min_vram_gb",
            "max_price_candidates",
            "max_provision_attempts",
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
        for path in (
            ".github/workflows/runpod-gpu-test.yml",
            ".github/workflows/CI_gpu.yml",
        ):
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
        for path in (
            ".github/workflows/runpod-gpu-test.yml",
            ".github/workflows/CI_gpu.yml",
        ):
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
                self.assertIn("-E 'test(pjrt_execution)'", text)
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
        self.assertEqual(minimum, (12, 4))
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

    def test_runpod_workflow_is_cache_reader_only(self) -> None:
        """No job that builds PR code or runs on the pod may write caches (#1403)."""

        text = read(".github/workflows/runpod-gpu-test.yml")
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
            '  CUDA_MIN_RUNTIME_VERSION: "12.4"',
            '  CUDA_RUNTIME_VERSION: "12.8"',
            '  CUTENSOR_VERSION: "2.6.0.4"',
            "  TENFERRO_CI_CACHE_ROOT: /opt/tenferro-ci",
        ):
            self.assertIn(env_line, consumer)
            self.assertIn(env_line, publisher)

    def test_gpu_retry_reuses_immutable_artifact(self) -> None:
        text = read(".github/workflows/runpod-gpu-test.yml")
        archive_block = text[
            text.index("  cuda-archive:") : text.index("  start-runpod:")
        ]
        reuse = archive_block.index("find_archive_artifact.py")
        build = archive_block.index("Build CUDA test archive")
        self.assertLess(reuse, build)
        # Every build-path step is skipped when the archives were restored
        # or reused, so a retry performs no Cargo compilation.
        self.assertEqual(
            archive_block.count(
                "if: steps.cuda_archive_cache.outputs.cache-hit != 'true' && steps.archive_reuse.outputs.reused != 'true'"
            ),
            5,
        )
        self.assertIn(
            "name: ${{ steps.archive_key.outputs.artifact_name }}", archive_block
        )
        run_gpu = text[
            text.index("  run-gpu-tests:") : text.index("  cleanup-runpod:")
        ]
        self.assertIn(
            "name: ${{ needs.cuda-archive.outputs.archive_artifact_name }}",
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
