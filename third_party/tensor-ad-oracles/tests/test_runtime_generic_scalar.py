import unittest
from types import SimpleNamespace

from generators import runtime


class RuntimeGenericScalarTests(unittest.TestCase):
    def _load_scalar_op(self, name: str):
        try:
            torch, cmi = runtime.import_scalar_generation_runtime()
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")
        op = next(row for row in cmi.op_db if row.name == name)
        return torch, op

    def _first_sample_with_kwargs(self, op, torch):
        for sample in op.sample_inputs("cpu", torch.float64, requires_grad=True):
            if sample.kwargs:
                return sample
        raise AssertionError(f"no kwargs sample found for {op.name}")

    def test_build_call_metadata_preserves_reduction_kwargs(self) -> None:
        torch, op = self._load_scalar_op("sum")

        sample = self._first_sample_with_kwargs(op, torch)

        op_args, op_kwargs = runtime.build_call_metadata(torch, sample)

        self.assertEqual(op_args, [])
        self.assertEqual(op_kwargs, sample.kwargs)

    def test_build_input_map_collects_tensor_kwargs(self) -> None:
        torch, op = self._load_scalar_op("nn.functional.prelu")

        sample = next(iter(op.sample_inputs("cpu", torch.float64, requires_grad=True)))
        spec = SimpleNamespace(
            op="nn_functional_prelu",
            upstream_name="nn.functional.prelu",
            gradcheck_wrapper=None,
        )

        inputs = runtime.build_input_map(torch, spec, sample)

        self.assertEqual(tuple(inputs), ("a", "b"))

    def test_call_upstream_op_rebuilds_generic_reduction_call(self) -> None:
        torch, op = self._load_scalar_op("sum")

        sample = self._first_sample_with_kwargs(op, torch)
        spec = SimpleNamespace(
            op="sum",
            upstream_name="sum",
            gradcheck_wrapper=None,
        )
        inputs = runtime.build_input_map(torch, spec, sample)
        op_args, op_kwargs = runtime.build_call_metadata(torch, sample)

        result = runtime.call_upstream_op(
            torch,
            spec,
            op,
            sample,
            inputs,
            op_args=op_args,
            op_kwargs=op_kwargs,
        )
        expected = op.op(sample.input, *sample.args, **sample.kwargs)

        self.assertTrue(torch.equal(result, expected))

    def test_dtype_name_supports_single_precision(self) -> None:
        torch, _ = self._load_scalar_op("sum")

        self.assertEqual(runtime.dtype_name(torch, torch.float32), "float32")
        self.assertEqual(runtime.dtype_name(torch, torch.complex64), "complex64")

    def test_build_call_metadata_canonicalizes_memory_format_kwargs(self) -> None:
        torch, op = self._load_scalar_op("double")
        sample = self._first_sample_with_kwargs(op, torch)

        op_args, op_kwargs = runtime.build_call_metadata(torch, sample)

        self.assertEqual(op_args, [])
        self.assertEqual(op_kwargs, {"memory_format": "contiguous_format"})

    def test_call_upstream_op_restores_stringified_dtype_kwargs(self) -> None:
        torch, op = self._load_scalar_op("prod")
        sample = self._first_sample_with_kwargs(op, torch)
        spec = SimpleNamespace(
            op="prod",
            upstream_name="prod",
            gradcheck_wrapper=None,
        )
        inputs = runtime.build_input_map(torch, spec, sample)
        op_args, op_kwargs = runtime.build_call_metadata(torch, sample)

        result = runtime.call_upstream_op(
            torch,
            spec,
            op,
            sample,
            inputs,
            op_args=op_args,
            op_kwargs=op_kwargs,
        )
        expected = op.op(sample.input, *sample.args, **sample.kwargs)

        self.assertEqual(op_kwargs, {"dtype": "float64"})
        self.assertTrue(torch.equal(result, expected))

    def test_call_upstream_op_accepts_native_memory_format_when_metadata_is_omitted(self) -> None:
        torch, op = self._load_scalar_op("double")
        sample = self._first_sample_with_kwargs(op, torch)
        spec = SimpleNamespace(
            op="double",
            upstream_name="double",
            gradcheck_wrapper=None,
        )
        inputs = runtime.build_input_map(torch, spec, sample)

        result = runtime.call_upstream_op(
            torch,
            spec,
            op,
            sample,
            inputs,
        )
        expected = op.op(sample.input, *sample.args, **sample.kwargs)

        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
