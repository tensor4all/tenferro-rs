import subprocess
import sys
import unittest
from pathlib import Path

import torch

from validators import replay

REPO_ROOT = Path(__file__).resolve().parents[1]


class DbReplayTests(unittest.TestCase):
    def test_validate_live_success_probe_requires_cross_oracle_jvp_agreement(self) -> None:
        with self.assertRaisesRegex(ValueError, "live PyTorch JVP and live FD-JVP disagree"):
            replay.validate_live_success_probe(
                torch,
                comparison={
                    "first_order": {"rtol": 1e-8, "atol": 1e-9},
                },
                direction={"a": torch.tensor([1.0], dtype=torch.float64)},
                cotangent={"value": torch.tensor([1.0], dtype=torch.float64)},
                pytorch_jvp={"value": torch.tensor([0.0], dtype=torch.float64)},
                pytorch_vjp={"a": torch.tensor([1.0], dtype=torch.float64)},
                fd_jvp={"value": torch.tensor([1.0], dtype=torch.float64)},
            )

    def test_validate_live_success_probe_requires_adjoint_consistency(self) -> None:
        with self.assertRaisesRegex(ValueError, "live probe failed adjoint consistency"):
            replay.validate_live_success_probe(
                torch,
                comparison={
                    "first_order": {"rtol": 1e-8, "atol": 1e-9},
                },
                direction={"a": torch.tensor([2.0], dtype=torch.float64)},
                cotangent={"value": torch.tensor([3.0], dtype=torch.float64)},
                pytorch_jvp={"value": torch.tensor([6.0], dtype=torch.float64)},
                pytorch_vjp={"a": torch.tensor([5.0], dtype=torch.float64)},
                fd_jvp={"value": torch.tensor([6.0], dtype=torch.float64)},
            )

    def test_validate_live_success_probe_requires_hvp_agreement_when_present(self) -> None:
        with self.assertRaisesRegex(ValueError, "live PyTorch HVP and live FD-HVP disagree"):
            replay.validate_live_success_probe(
                torch,
                comparison={
                    "first_order": {"rtol": 1e-8, "atol": 1e-9},
                    "second_order": {"rtol": 1e-8, "atol": 1e-9},
                },
                direction={"a": torch.tensor([1.0], dtype=torch.float64)},
                cotangent={"value": torch.tensor([1.0], dtype=torch.float64)},
                pytorch_jvp={"value": torch.tensor([1.0], dtype=torch.float64)},
                pytorch_vjp={"a": torch.tensor([1.0], dtype=torch.float64)},
                fd_jvp={"value": torch.tensor([1.0], dtype=torch.float64)},
                pytorch_hvp={"a": torch.tensor([0.0], dtype=torch.float64)},
                fd_hvp={"a": torch.tensor([1.0], dtype=torch.float64)},
            )

    def test_check_replay_script_succeeds_against_published_case_tree(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/check_replay.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=(completed.stdout + completed.stderr).strip(),
        )
        self.assertIn("replay_checked=", completed.stdout)
