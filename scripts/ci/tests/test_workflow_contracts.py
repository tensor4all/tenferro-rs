import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read(path: str) -> str:
    return (ROOT / path).read_text()


class WorkflowContractTests(unittest.TestCase):
    def test_fast_ci_uses_shared_policy_and_profiles(self) -> None:
        text = read(".github/workflows/ci.yml")
        self.assertIn("python3 scripts/ci/change_policy.py", text)
        self.assertIn("python3 scripts/ci/run_profile.py blas-inject", text)
        self.assertIn("python3 scripts/ci/run_profile.py coverage", text)
        self.assertIn("python3 scripts/ci/run_profile.py docs", text)
        self.assertIn("name: CI configuration checks", text)

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
            text.index("      - name: Create RunPod pod") : text.index(
                "      - name: Wait for org runner to come online"
            )
        ]
        self.assertIn("python3 -m scripts.ci.runpod_client create", create)
        self.assertNotIn("python3 scripts/ci/runpod_client.py", create)
        self.assertNotIn("for attempt in $(seq 1 5)", create)
        self.assertNotIn("curl -sS", create)

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
        # Material build inputs only (#1403): unrelated workflow or RunPod
        # configuration edits must not invalidate the archive key.
        self.assertNotIn("runpod_config.json", key_line)
        self.assertNotIn(".github/workflows", key_line)
        self.assertIn("rust${rustc_version}", key_line)
        for material in (
            "tenferro-rs/Cargo.lock",
            "tenferro-rs/**/Cargo.toml",
            "tenferro-rs/**/src/**",
            "tenferro-rs/**/tests/**",
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
        # its checkouts must not override the ref.
        self.assertNotIn("ref:", text)
        self.assertIn("actions: write", text)

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
            "prefix-key: v7-rust-cuda-pjrt-ci-ubuntu22-ptx",
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
