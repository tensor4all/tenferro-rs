import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.ci.change_policy import ChangeClass, _changed_paths, classify_paths


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "change_policy.py"


class ChangePolicyTests(unittest.TestCase):
    def test_repository_prose_is_docs_only(self) -> None:
        # The instruction files and the contribution workflows are prose about
        # how work is done; they cannot change what any lane executes, so they
        # must not reserve the paid GPU lane.
        for path in (
            "AGENTS.md",
            "REPOSITORY_RULES.md",
            "PERFORMANCE_TIPS.md",
            "ai/contribution-workflows/release-publish.md",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.DOCS_ONLY)
                self.assertTrue(policy.run_docs)
                self.assertFalse(policy.run_rust)
                self.assertFalse(policy.run_gpu)

    def test_docs_only_runs_docs_without_rust_or_gpu(self) -> None:
        policy = classify_paths(["docs/guides/cpu.md", "README.md"])
        self.assertIs(policy.change_class, ChangeClass.DOCS_ONLY)
        self.assertTrue(policy.run_docs)
        self.assertFalse(policy.run_rust)
        self.assertFalse(policy.run_extensions)
        self.assertFalse(policy.run_gpu)
        self.assertFalse(policy.run_macos)

    def test_markdown_under_docs_is_prose_even_in_the_tutorial_tree(self) -> None:
        # The tutorial tree holds runnable binaries, but a README there is prose:
        # editing one must not reserve the paid GPU lane (#2376).
        for path in (
            "docs/tutorial-code/README.md",
            ".github/releases/v0.9.0.md",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.DOCS_ONLY)
                self.assertTrue(policy.run_docs)
                self.assertFalse(policy.run_rust)
                self.assertFalse(policy.run_gpu)

    def test_non_markdown_in_the_tutorial_tree_stays_code(self) -> None:
        # Only the prose widens; the manifest and the binaries still build.
        for path in (
            "docs/tutorial-code/Cargo.toml",
            "docs/tutorial-code/notes.txt",
            ".github/releases/payload.json",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.CODE)
                self.assertTrue(policy.run_rust)
                self.assertTrue(policy.run_gpu)

    def test_ci_and_docs_runs_both_lightweight_suites(self) -> None:
        policy = classify_paths(
            [".github/workflows/ci.yml", "docs/worklogs/ci.md"]
        )
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertTrue(policy.run_ci_config)
        self.assertTrue(policy.run_docs)
        self.assertFalse(policy.run_rust)
        self.assertFalse(policy.run_macos)

    def test_macos_control_plane_requires_macos(self) -> None:
        for path in (
            ".github/workflows/ci-pr-workspace-tests.yml",
            "scripts/ci/change_policy.py",
            "scripts/ci/run_profile.py",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
                self.assertTrue(policy.run_ci_config)
                self.assertTrue(policy.run_macos)

    def test_inventory_checker_uses_ci_config_lane(self) -> None:
        policy = classify_paths(["scripts/check-public-boundary-inventory.py"])
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertTrue(policy.run_ci_config)
        self.assertFalse(policy.run_docs)
        self.assertFalse(policy.run_rust)

    def test_review_gate_python_scripts_use_ci_config_lane(self) -> None:
        """`review_bot.yml` runs the reviewer; no pod ever imports either file.

        Both files are Python, and the device suite the paid lane runs cannot
        observe them: `.github/workflows/review_bot.yml` executes
        `repository-rules-review.py` from the trusted base revision, and
        `test-repository-rules-review.py` is the fixture suite that exercises
        it. A Python-only change to either therefore belongs to the CI
        configuration lane and must not reserve a pod (#2512).
        """
        for path in (
            "scripts/repository-rules-review.py",
            "scripts/test-repository-rules-review.py",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
                self.assertTrue(policy.run_ci_config)
                self.assertFalse(policy.run_rust)
                self.assertFalse(policy.run_gpu)
                self.assertFalse(policy.run_macos)

    def test_unrelated_ci_change_does_not_require_macos(self) -> None:
        policy = classify_paths([".github/workflows/docs.yml"])
        self.assertFalse(policy.run_macos)

    def test_gpu_device_control_plane_requires_gpu(self) -> None:
        """A diff that can change what the pod executes is proven on a pod."""
        for path in (
            "scripts/ci/runpod_client.py",
            "scripts/ci/runpod_http.py",
            "scripts/ci/runpod_config.json",
            ".github/workflows/runpod-gpu-test.yml",
            ".github/workflows/runpod-gpu-execute.yml",
            ".github/workflows/ci-cache-publish.yml",
            "scripts/ci/find_archive_artifact.py",
            "scripts/ci/gpu_test_partition.py",
            "scripts/ci/gpu_nextest.toml",
            "scripts/ci/cuda_test_partition.tsv",
            "scripts/ci/pjrt_test_partition.tsv",
            "scripts/ci/install_cuda_toolkit_hosted.sh",
            "scripts/ci/install_cuda_runtime_tree.sh",
            ".github/workflows/runpod-gpu-runtime.yml",
            "scripts/ci/prepare_gpu_execution_payload.sh",
            "scripts/ci/check_cuda_headers.py",
            "scripts/ci/install_cutensor.sh",
            "scripts/ci/runpod_provision.py",
            "scripts/ci/runpod_setup_watchdog.py",
            "scripts/ci/runpod_lifecycle.py",
            "scripts/ci/cuda_smoke_test.py",
            "scripts/ci/change_policy.py",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
                self.assertTrue(policy.run_ci_config)
                self.assertTrue(policy.run_gpu)

    def test_gpu_host_side_control_plane_does_not_require_gpu(self) -> None:
        """The host half of the paid lane is never observed by the device run.

        A pod runs the device suite; it cannot exercise the required-gate
        contract, the reuse lookup, the price table, or the orphaned-pod
        reaper. Those paths have their own unit tests, and spending a pod on
        them is what this list exists to prevent (#2163).
        """
        for path in (
            ".github/workflows/CI_gpu.yml",
            ".github/workflows/local-gpu-gate.yml",
            ".github/workflows/runpod-reap.yml",
            "scripts/ci/gpu_gate_reuse.py",
            "scripts/ci/local_gpu_gate.py",
            "scripts/ci/runpod_pricing.py",
            "scripts/ci/runpod_cost.py",
            "scripts/ci/runpod_reap.py",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
                self.assertTrue(policy.run_ci_config)
                self.assertFalse(policy.run_gpu)

    def test_unrelated_ci_change_does_not_require_gpu(self) -> None:
        policy = classify_paths([".github/workflows/docs.yml"])
        self.assertIs(policy.change_class, ChangeClass.CI_ONLY)
        self.assertTrue(policy.run_ci_config)
        self.assertFalse(policy.run_gpu)

    def test_tutorial_workspace_member_and_shared_dependencies_require_gpu(self) -> None:
        for path in (
            "docs/tutorial-code/src/bin/cuda_tutorial.rs",
            "docs/tutorial-code/Cargo.toml",
            "docs/tutorial-code/build.rs",
            "crates/tenferro-tensor/src/types.rs",
            "crates/tenferro-runtime/src/lib.rs",
            "crates/tenferro-cpu/src/lib.rs",
            "crates/tenferro-ad/src/lib.rs",
            "Cargo.toml",
            "Cargo.lock",
        ):
            with self.subTest(path=path):
                policy = classify_paths([path])
                self.assertIs(policy.change_class, ChangeClass.CODE)
                self.assertTrue(policy.run_rust)
                self.assertTrue(policy.run_gpu)

    def test_unknown_and_empty_diffs_fall_back_to_code(self) -> None:
        for paths in ([], ["new-top-level-policy.toml"]):
            with self.subTest(paths=paths):
                policy = classify_paths(paths)
                self.assertIs(policy.change_class, ChangeClass.CODE)
                self.assertTrue(policy.run_rust)
                self.assertTrue(policy.run_extensions)
                self.assertTrue(policy.run_gpu)
                self.assertTrue(policy.run_macos)

    def test_push_to_main_forces_comprehensive_lanes(self) -> None:
        policy = classify_paths(["README.md"], event="push")
        self.assertIs(policy.change_class, ChangeClass.CODE)
        self.assertTrue(policy.run_rust)
        self.assertTrue(policy.run_extensions)
        self.assertTrue(policy.run_docs)
        self.assertTrue(policy.run_ci_config)
        self.assertTrue(policy.run_macos)

    def test_paths_are_normalized_and_deduplicated(self) -> None:
        policy = classify_paths([" ./README.md ", "README.md", ""])
        self.assertIs(policy.change_class, ChangeClass.DOCS_ONLY)
        self.assertEqual(policy.reasons, ("docs: README.md",))

    def test_cli_writes_stable_github_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            env = os.environ | {"GITHUB_OUTPUT": str(output)}
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--path", "README.md"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["classification"], "docs-only")
            values = dict(
                line.split("=", 1) for line in output.read_text().splitlines()
            )
            self.assertEqual(values["classification"], "docs-only")
            self.assertEqual(values["run_docs"], "true")
            self.assertEqual(values["run_rust"], "false")
            self.assertEqual(values["run_macos"], "false")
            self.assertIn("README.md", values["reason"])

    def test_cli_classifies_deleted_documentation_as_docs_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo,
                check=True,
            )
            docs = repo / "docs"
            docs.mkdir()
            guide = docs / "guide.md"
            guide.write_text("guide\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-m", "add guide"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            base = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            guide.unlink()
            subprocess.run(["git", "add", "-u"], cwd=repo, check=True)
            subprocess.run(
                ["git", "commit", "-m", "remove guide"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base",
                    base,
                    "--head",
                    head,
                ],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(result.stdout)["classification"], "docs-only")


class RustCommentPolicyTests(unittest.TestCase):
    """A `.rs` diff that only edits documentation comments needs no GPU pod."""

    LIB = "crates/demo/src/lib.rs"
    INTEGRATION = "crates/demo/tests/integration.rs"
    UNIT_TEST = "crates/demo/src/demo_test.rs"
    COUNTER = "crates/demo/src/counter.rs"
    DOC_LIB = (
        "/// Returns one.\n"
        "pub fn demo() -> i32 {\n"
        "    let mut value = 1;\n"
        "    value\n"
        "}\n"
    )

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        self.git("init", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test User")
        self.write(
            self.LIB,
            "pub fn demo() -> i32 {\n    let mut value = 1;\n    value\n}\n",
        )
        self.write(self.INTEGRATION, "fn helper() {}\n")
        self.write(self.UNIT_TEST, "fn helper() {}\n")
        self.write(
            self.COUNTER, "pub fn bump(ptr: &mut i32) {\n    *ptr += 1;\n}\n"
        )
        self.base = self.commit("baseline")
        self.head = self.base

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def write(self, path: str, text: str) -> None:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD").strip()

    def edit(self, path: str, text: str) -> None:
        self.write(path, text)
        self.head = self.commit("edit")

    def classify(
        self,
        *paths: str,
        base: str | None = None,
        head: str | None = None,
    ):
        return classify_paths(
            list(paths),
            base=self.base if base is None else base,
            head=self.head if head is None else head,
            cwd=str(self.repo),
        )

    def test_doc_comment_only_edit_skips_the_paid_gpu_lane(self) -> None:
        self.edit(
            self.LIB,
            "//! Demo crate.\n"
            "/// Returns one.\n"
            "///\n"
            "/// ```\n"
            "/// assert_eq!(demo(), 1);\n"
            "/// ```\n"
            "pub fn demo() -> i32 {\n"
            "    let mut value = 1;\n"
            "    value\n"
            "}\n",
        )
        policy = self.classify(self.LIB)
        self.assertIs(policy.change_class, ChangeClass.CODE)
        self.assertFalse(policy.run_gpu)
        # Doc comments still hold doctests, so every hosted lane keeps running.
        self.assertTrue(policy.run_rust)
        self.assertTrue(policy.run_docs)
        self.assertTrue(policy.run_macos)
        self.assertIn("docs-only Rust comments", "; ".join(policy.reasons))
        self.assertIn(self.LIB, "; ".join(policy.reasons))

    def test_doc_comment_plus_one_code_line_keeps_the_paid_gpu_lane(self) -> None:
        self.edit(
            self.LIB,
            "/// Returns two.\n"
            "pub fn demo() -> i32 {\n"
            "    let value = 2;\n"
            "    value\n"
            "}\n",
        )
        policy = self.classify(self.LIB)
        self.assertTrue(policy.run_gpu)
        self.assertIn("code or unknown paths", "; ".join(policy.reasons))

    def test_comment_only_edit_in_a_test_target_keeps_the_paid_gpu_lane(
        self,
    ) -> None:
        for path in (self.INTEGRATION, self.UNIT_TEST):
            with self.subTest(path=path):
                self.edit(path, "/// Explains the helper.\nfn helper() {}\n")
                policy = self.classify(path)
                self.assertTrue(policy.run_gpu)

    def test_blank_lines_and_block_comments_are_documentation(self) -> None:
        self.edit(
            self.LIB,
            "/**\n"
            " * Returns the constant one.\n"
            " *\n"
            " * Second paragraph, still prose.\n"
            " */\n"
            "pub fn demo() -> i32 {\n"
            "    let mut value = 1;\n"
            "    value\n"
            "}\n",
        )
        policy = self.classify(self.LIB)
        self.assertFalse(policy.run_gpu)

    def test_blank_line_only_edit_keeps_hosted_lanes_and_skips_the_pod(
        self,
    ) -> None:
        self.edit(
            self.LIB,
            "pub fn demo() -> i32 {\n    let mut value = 1;\n\n    value\n}\n",
        )
        # Adding a blank line is whitespace-only, but it is not documentation
        # either: the guard accepts it, and every other lane still runs.
        policy = self.classify(self.LIB)
        self.assertFalse(policy.run_gpu)
        self.assertTrue(policy.run_rust)

    def test_doc_attribute_line_is_documentation(self) -> None:
        self.edit(
            self.LIB,
            '#[doc = "Returns the constant one."]\n'
            "pub fn demo() -> i32 {\n"
            "    let mut value = 1;\n"
            "    value\n"
            "}\n",
        )
        policy = self.classify(self.LIB)
        self.assertFalse(policy.run_gpu)

    def test_non_rustdoc_comment_keeps_the_paid_gpu_lane(self) -> None:
        self.edit(
            self.LIB,
            "// Not rustdoc, so it can sit next to code that changed meaning.\n"
            "pub fn demo() -> i32 {\n"
            "    let mut value = 1;\n"
            "    value\n"
            "}\n",
        )
        policy = self.classify(self.LIB)
        self.assertTrue(policy.run_gpu)

    def test_dereference_line_is_not_a_block_comment_body(self) -> None:
        self.edit(
            self.COUNTER, "pub fn bump(ptr: &mut i32) {\n    *ptr += 2;\n}\n"
        )
        policy = self.classify(self.COUNTER)
        self.assertTrue(policy.run_gpu)

    def test_git_failure_falls_back_to_the_paid_gpu_lane(self) -> None:
        self.edit(self.LIB, self.DOC_LIB)
        policy = self.classify(self.LIB, base="0" * 40)
        self.assertTrue(policy.run_gpu)

    def test_missing_endpoints_keep_the_paid_gpu_lane(self) -> None:
        self.edit(self.LIB, self.DOC_LIB)
        policy = classify_paths([self.LIB])
        self.assertTrue(policy.run_gpu)

    def test_doc_comment_edit_beside_a_gpu_control_plane_file_keeps_the_pod(
        self,
    ) -> None:
        self.edit(self.LIB, self.DOC_LIB)
        policy = self.classify(self.LIB, "scripts/ci/runpod_pricing.py")
        self.assertFalse(policy.run_gpu)
        policy = self.classify(self.LIB, "scripts/ci/runpod_client.py")
        self.assertTrue(policy.run_gpu)
        self.assertIn("ci: scripts/ci/runpod_client.py", policy.reasons)

    def test_doc_comment_edit_beside_prose_still_skips_the_paid_gpu_lane(
        self,
    ) -> None:
        self.edit(self.LIB, self.DOC_LIB)
        (self.repo / "README.md").write_text("readme\n")
        self.head = self.commit("docs")
        policy = self.classify(self.LIB, "README.md")
        self.assertIs(policy.change_class, ChangeClass.CODE)
        self.assertFalse(policy.run_gpu)
        self.assertTrue(policy.run_docs)

    def test_cli_reads_hunks_when_paths_and_endpoints_are_given(self) -> None:
        self.edit(self.LIB, self.DOC_LIB)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--path",
                self.LIB,
                "--base",
                self.base,
                "--head",
                self.head,
            ],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["run_gpu"], False)
        self.assertEqual(payload["run_rust"], True)
        self.assertIn("docs-only Rust comments", payload["reason"])


class StaleBaseTests(unittest.TestCase):
    """A base branch that moved past the fork point must not add its files.

    `--base` carries the base branch tip (`github.event.pull_request.base.sha`),
    not the fork point. Comparing the two trees directly therefore also reports
    every path that landed on the base branch *after* the fork, as if the pull
    request had removed it. Reading the diff from the merge base is what keeps
    an untouched file out: a late landing on a GPU control-plane path otherwise
    reserves a paid pod for a pull request that never touched it.
    """

    PR_FILE = "README.md"
    LATE_MAIN_CI_FILE = "scripts/ci/runpod_provision.py"
    LATE_MAIN_CODE_FILE = "crates/demo/src/lib.rs"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def git(self, repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def commit(self, repo: Path, message: str) -> str:
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-q", "-m", message)
        return self.git(repo, "rev-parse", "HEAD").strip()

    def scenario(self, name: str, late_path: str) -> tuple[Path, str, str]:
        """Build a fork, a prose-only pull request, and a late base landing."""
        repo = Path(self.temp_dir.name) / name
        repo.mkdir(parents=True)
        self.git(repo, "init", "-q", "-b", "main")
        self.git(repo, "config", "user.email", "test@example.com")
        self.git(repo, "config", "user.name", "Test User")
        (repo / "docs").mkdir()
        (repo / "docs" / "guide.md").write_text("guide\n")
        self.commit(repo, "fork point")

        self.git(repo, "checkout", "-q", "-b", "pr")
        (repo / self.PR_FILE).write_text("pull request\n")
        head = self.commit(repo, "pr: prose only")

        self.git(repo, "checkout", "-q", "main")
        late = repo / late_path
        late.parent.mkdir(parents=True, exist_ok=True)
        late.write_text("landed on the base branch after the fork\n")
        base = self.commit(repo, "base: late landing")

        self.git(repo, "checkout", "-q", "pr")
        return repo, base, head

    def classify(self, repo: Path, base: str, head: str) -> dict:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--base", base, "--head", head],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_late_gpu_control_plane_landing_reserves_no_pod(self) -> None:
        repo, base, head = self.scenario("ci", self.LATE_MAIN_CI_FILE)
        payload = self.classify(repo, base, head)
        self.assertEqual(payload["classification"], "docs-only")
        self.assertFalse(payload["run_gpu"])
        self.assertFalse(payload["run_ci_config"])
        self.assertNotIn(self.LATE_MAIN_CI_FILE, payload["reason"])

    def test_late_code_landing_does_not_widen_the_class(self) -> None:
        repo, base, head = self.scenario("code", self.LATE_MAIN_CODE_FILE)
        payload = self.classify(repo, base, head)
        self.assertEqual(payload["classification"], "docs-only")
        self.assertFalse(payload["run_gpu"])
        self.assertFalse(payload["run_rust"])
        self.assertNotIn(self.LATE_MAIN_CODE_FILE, payload["reason"])

    def test_changed_paths_are_read_from_the_fork_point(self) -> None:
        repo, base, head = self.scenario("paths", self.LATE_MAIN_CODE_FILE)
        self.assertEqual(
            _changed_paths(base, head, cwd=str(repo)), [self.PR_FILE]
        )
        self.assertIsNone(_changed_paths("0" * 40, head, cwd=str(repo)))

    def test_unresolvable_split_point_fails_closed(self) -> None:
        """An unknown base keeps full validation; it never guesses a lane."""
        repo, _, head = self.scenario("missing", self.LATE_MAIN_CI_FILE)
        payload = self.classify(repo, "0" * 40, head)
        self.assertEqual(payload["classification"], "code")
        self.assertTrue(payload["run_gpu"])

    def test_shallow_history_without_the_fork_point_fails_closed(self) -> None:
        """A shallow clone has both endpoints but no merge base to compare."""
        repo, base, head = self.scenario("shallow", self.LATE_MAIN_CI_FILE)
        clone = Path(self.temp_dir.name) / "shallow-clone"
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(clone)],
            check=True,
            capture_output=True,
        )
        for revision in (base, head):
            subprocess.run(
                ["git", "fetch", "-q", "--depth", "1", "origin", revision],
                cwd=clone,
                check=True,
                capture_output=True,
            )
        payload = self.classify(clone, base, head)
        self.assertEqual(payload["classification"], "code")
        self.assertTrue(payload["run_gpu"])


class LocalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        (self.repo / "scripts" / "ci").mkdir(parents=True)
        (self.repo / "scripts" / "lib").mkdir(parents=True)
        shutil.copy2(ROOT / "scripts" / "check-pr-fast.sh", self.repo / "scripts")
        shutil.copy2(
            ROOT / "scripts" / "ci" / "change_policy.py",
            self.repo / "scripts" / "ci",
        )
        shutil.copy2(
            ROOT / "scripts" / "ci" / "run_profile.py",
            self.repo / "scripts" / "ci",
        )
        shutil.copy2(
            ROOT / "scripts" / "lib" / "python.sh",
            self.repo / "scripts" / "lib",
        )
        # The gate audits execution-engine entry surfaces on every code change.
        # The fixture has no crates, so it gets an allowlist of the same
        # mechanisms with no recorded sites (the real entries would all read as
        # stale).
        shutil.copy2(ROOT / "scripts" / "audit-engine-entry.py", self.repo / "scripts")
        # The gate asserts one version per dependency identity on every non-docs
        # change; the fixture's stub cargo prints no tree, so it passes.
        shutil.copy2(ROOT / "scripts" / "check-single-versions.sh", self.repo / "scripts")
        allowlist = json.loads(
            (ROOT / "scripts" / "engine-entry-allowlist.json").read_text(encoding="utf-8")
        )
        (self.repo / "scripts" / "engine-entry-allowlist.json").write_text(
            json.dumps({mechanism: {} for mechanism in allowlist}), encoding="utf-8"
        )
        bin_dir = self.repo / "bin"
        bin_dir.mkdir()
        cargo = bin_dir / "cargo"
        cargo.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$*\" >> \"$CARGO_MARKER\"\n"
        )
        cargo.chmod(0o755)
        subprocess.run(["git", "init", "-b", "test-branch"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.repo,
            check=True,
        )
        (self.repo / "README.md").write_text("baseline\n")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", "baseline"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        self.marker = self.repo / "cargo-calls"
        self.env = os.environ | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CARGO_MARKER": str(self.marker),
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_gate(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "scripts/check-pr-fast.sh",
                "--base",
                "HEAD",
                "--no-fetch",
                "--skip-doc-snippets",
                *args,
            ],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def write_change(self, path: str) -> None:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("changed\n")

    def test_code_change_requires_a_focused_command(self) -> None:
        self.write_change("src/lib.rs")
        result = self.run_gate("--coverage-reviewed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("focused verification command required", result.stderr)

    def test_code_change_runs_focused_command_incrementally(self) -> None:
        self.write_change("src/lib.rs")
        result = self.run_gate("--coverage-reviewed", "--test", "true")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("classification: code", result.stdout)
        self.assertTrue(self.marker.exists(), "code changes should run cargo fmt")

    def test_clean_base_head_is_a_valid_no_change_gate_under_bin_bash(self) -> None:
        compatible_python_dir = Path(sys.executable).resolve().parent
        env = self.env | {
            "PATH": f"{compatible_python_dir}:{self.env['PATH']}",
        }
        result = subprocess.run(
            [
                "/bin/bash",
                "scripts/check-pr-fast.sh",
                "--base",
                "HEAD",
                "--no-fetch",
                "--skip-doc-snippets",
                "--coverage-reviewed",
                "--test",
                "true",
            ],
            cwd=self.repo,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("changed files: none", result.stdout)

    def test_docs_only_change_skips_cargo_and_coverage_acknowledgement(self) -> None:
        self.write_change("docs/guide.md")
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("classification: docs-only", result.stdout)
        self.assertFalse(self.marker.exists(), "docs-only must not invoke Cargo")

    def test_untracked_docs_with_trailing_whitespace_fail(self) -> None:
        target = self.repo / "docs" / "guide.md"
        target.parent.mkdir(parents=True)
        target.write_text("trailing whitespace   \n")
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("trailing whitespace", result.stdout + result.stderr)

    def test_ci_only_change_requires_a_focused_command(self) -> None:
        self.write_change("scripts/ci/example.py")
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("focused verification command required", result.stderr)

    def test_ci_only_change_accepts_an_explicit_ci_profile(self) -> None:
        self.write_change("scripts/ci/example.py")
        result = self.run_gate(
            "--ci-profile", "ci-config", "--ci-profile-dry-run"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("python3 -m unittest discover", result.stdout)


if __name__ == "__main__":
    unittest.main()
