import subprocess
import unittest

from scripts.ci.recover_runpod_pr import (
    build_dispatch_command,
    recover_pr,
)


class RecoverRunPodPrTests(unittest.TestCase):
    def test_dispatch_always_uses_trusted_main(self) -> None:
        self.assertEqual(
            build_dispatch_command(1379, wait=False),
            [
                "gh",
                "workflow",
                "run",
                "runpod-gpu-test.yml",
                "--ref",
                "main",
                "-f",
                "pr_number=1379",
            ],
        )

    def test_invalid_pr_numbers_are_rejected(self) -> None:
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "positive"
            ):
                build_dispatch_command(value, wait=False)

    def test_recovery_authenticates_dispatches_and_watches_returned_run(self) -> None:
        commands: list[list[str]] = []

        def runner(
            command: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            stdout = (
                "https://github.com/tensor4all/tenferro-rs/actions/runs/12345\n"
                if command[1:3] == ["workflow", "run"]
                else ""
            )
            return subprocess.CompletedProcess(command, 0, stdout, "")

        url = recover_pr(1379, wait=True, runner=runner)
        self.assertEqual(
            url,
            "https://github.com/tensor4all/tenferro-rs/actions/runs/12345",
        )
        self.assertEqual(commands[0], ["gh", "auth", "status"])
        self.assertEqual(commands[-1], ["gh", "run", "watch", "12345", "--exit-status"])

    def test_missing_run_url_is_an_error(self) -> None:
        def runner(
            command: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 0, "dispatched\n", "")

        with self.assertRaisesRegex(RuntimeError, "run URL"):
            recover_pr(1379, wait=False, runner=runner)


if __name__ == "__main__":
    unittest.main()
