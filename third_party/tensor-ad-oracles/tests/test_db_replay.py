import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch

from validators import replay

class DbReplayTests(unittest.TestCase):
    def test_find_candidate_samples_allows_small_input_drift(self) -> None:
        spec = object()
        sample = object()
        record_inputs = {"a": torch.tensor([1.0], dtype=torch.float64)}
        sample_inputs = {"a": torch.tensor([1.0 + 5e-9], dtype=torch.float64)}

        with (
            mock.patch.object(replay, "import_generation_runtime", return_value=(None, object())),
            mock.patch.object(replay, "sample_inputs_for_spec", return_value=[sample]),
            mock.patch.object(replay, "build_input_map", return_value=sample_inputs),
            mock.patch.object(replay, "build_call_metadata", return_value=([], {})),
        ):
            candidates = replay._find_candidate_samples(
                torch,
                spec,
                record_inputs,
                comparison={"first_order": {"rtol": 1e-8, "atol": 1e-9}},
            )

        self.assertEqual(candidates, [sample])

    def test_find_candidate_samples_skips_shape_mismatches(self) -> None:
        spec = object()
        sample = object()
        record_inputs = {"a": torch.ones((2, 2), dtype=torch.float64)}
        sample_inputs = {"a": torch.ones((2, 0), dtype=torch.float64)}

        with (
            mock.patch.object(replay, "import_generation_runtime", return_value=(None, object())),
            mock.patch.object(replay, "sample_inputs_for_spec", return_value=[sample]),
            mock.patch.object(replay, "build_input_map", return_value=sample_inputs),
            mock.patch.object(replay, "build_call_metadata", return_value=([], {})),
        ):
            candidates = replay._find_candidate_samples(
                torch,
                spec,
                record_inputs,
                comparison={"first_order": {"rtol": 1e-8, "atol": 1e-9}},
            )

        self.assertEqual(candidates, [])

    def test_find_candidate_samples_uses_scalar_runtime_and_record_dtype(self) -> None:
        spec = SimpleNamespace(inventory_kind="scalar")
        sample = object()
        record_inputs = {"a": torch.tensor([1.0], dtype=torch.float32)}

        with (
            mock.patch.object(
                replay,
                "import_scalar_generation_runtime",
                return_value=(torch, object()),
            ) as import_scalar_runtime,
            mock.patch.object(replay, "sample_inputs_for_spec", return_value=[sample]) as sample_inputs,
            mock.patch.object(replay, "build_input_map", return_value=record_inputs),
            mock.patch.object(replay, "build_call_metadata", return_value=([], {})),
        ):
            candidates = replay._find_candidate_samples(
                torch,
                spec,
                record_inputs,
                comparison={"first_order": {"rtol": 1e-8, "atol": 1e-9}},
                dtype_name="float32",
                op_args=[],
                op_kwargs={},
            )

        self.assertEqual(candidates, [sample])
        import_scalar_runtime.assert_called_once_with()
        sample_inputs.assert_called_once()
        self.assertEqual(sample_inputs.call_args.kwargs["dtype"], torch.float32)

    def test_find_candidate_samples_treats_nan_payloads_as_matching(self) -> None:
        spec = SimpleNamespace(inventory_kind="scalar")
        sample = object()
        record_inputs = {"a": torch.tensor([1.0, float("nan")], dtype=torch.float32)}
        sample_inputs = {"a": torch.tensor([1.0, float("nan")], dtype=torch.float32)}

        with (
            mock.patch.object(
                replay,
                "import_scalar_generation_runtime",
                return_value=(torch, object()),
            ),
            mock.patch.object(replay, "sample_inputs_for_spec", return_value=[sample]),
            mock.patch.object(replay, "build_input_map", return_value=sample_inputs),
            mock.patch.object(replay, "build_call_metadata", return_value=([], {})),
        ):
            candidates = replay._find_candidate_samples(
                torch,
                spec,
                record_inputs,
                comparison={"first_order": {"rtol": 1e-4, "atol": 1e-6}},
                dtype_name="float32",
                op_args=[],
                op_kwargs={},
            )

        self.assertEqual(candidates, [sample])

    def test_find_candidate_samples_rejects_metadata_mismatch(self) -> None:
        spec = SimpleNamespace(inventory_kind="scalar")
        sample = object()
        record_inputs = {"a": torch.tensor([1.0], dtype=torch.float64)}

        with (
            mock.patch.object(
                replay,
                "import_scalar_generation_runtime",
                return_value=(torch, object()),
            ),
            mock.patch.object(replay, "sample_inputs_for_spec", return_value=[sample]),
            mock.patch.object(replay, "build_input_map", return_value=record_inputs),
            mock.patch.object(replay, "build_call_metadata", return_value=([], {"dtype": "float64"})),
        ):
            candidates = replay._find_candidate_samples(
                torch,
                spec,
                record_inputs,
                comparison={"first_order": {"rtol": 1e-8, "atol": 1e-9}},
                dtype_name="float64",
                op_args=[],
                op_kwargs={"dtype": "complex128"},
            )

        self.assertEqual(candidates, [])

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

    def test_validate_live_success_probe_promotes_mixed_scalar_dtypes_for_adjoint_check(self) -> None:
        replay.validate_live_success_probe(
            torch,
            comparison={
                "first_order": {"rtol": 1e-8, "atol": 1e-9},
            },
            direction={"a": torch.tensor([3.0], dtype=torch.float32)},
            cotangent={"value": torch.tensor([2.0], dtype=torch.float64)},
            pytorch_jvp={"value": torch.tensor([3.0], dtype=torch.float64)},
            pytorch_vjp={"a": torch.tensor([2.0], dtype=torch.float32)},
            fd_jvp={"value": torch.tensor([3.0], dtype=torch.float64)},
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

    def test_replay_case_file_accepts_generated_scalar_case_family(self) -> None:
        try:
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            cases_root = Path(tmpdir)
            from generators import pytorch_v1

            generated = pytorch_v1.materialize_case_family(
                "abs",
                "identity",
                limit=1,
                cases_root=cases_root,
            )
            result = replay.replay_case_file(generated)

            self.assertEqual(result.checked, 4, msg=result.failures)
            self.assertEqual(result.failures, [])

    def test_replay_case_file_reuses_sample_inputs_for_same_spec_and_dtype(self) -> None:
        try:
            import expecttest  # noqa: F401
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            cases_root = Path(tmpdir)
            from generators import pytorch_v1

            generated = pytorch_v1.materialize_case_family(
                "solve",
                "identity",
                limit=3,
                cases_root=cases_root,
            )
            real_sample_inputs = replay.sample_inputs_for_spec
            call_count = 0

            def counting_sample_inputs(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return real_sample_inputs(*args, **kwargs)

            with mock.patch.object(replay, "sample_inputs_for_spec", side_effect=counting_sample_inputs):
                result = replay.replay_case_file(generated)

            spec = pytorch_v1.build_case_spec_index()[("solve", "identity")]
            self.assertEqual(result.failures, [])
            self.assertEqual(result.checked, 3 * len(spec.supported_dtype_names))
            self.assertEqual(call_count, len(spec.supported_dtype_names))
