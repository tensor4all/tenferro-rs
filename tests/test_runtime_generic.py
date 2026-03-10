import unittest
from types import SimpleNamespace

from generators import runtime


class RuntimeGenericTests(unittest.TestCase):
    def test_build_input_map_keeps_two_tensor_inputs_as_a_b(self) -> None:
        try:
            torch, linalg = runtime.import_generation_runtime()
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        op = next(row for row in linalg.op_db if row.name == "linalg.solve")
        sample = next(iter(op.sample_inputs("cpu", torch.float64, requires_grad=True)))
        spec = SimpleNamespace(
            op="solve",
            upstream_name="linalg.solve",
            gradcheck_wrapper=None,
        )

        inputs = runtime.build_input_map(torch, spec, sample)

        self.assertEqual(tuple(inputs), ("a", "b"))

    def test_build_input_map_flattens_tensor_list_inputs_in_order(self) -> None:
        try:
            torch, linalg = runtime.import_generation_runtime()
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        op = next(row for row in linalg.op_db if row.name == "linalg.multi_dot")
        sample = next(iter(op.sample_inputs("cpu", torch.float64, requires_grad=True)))
        spec = SimpleNamespace(
            op="multi_dot",
            upstream_name="linalg.multi_dot",
            gradcheck_wrapper=None,
        )

        inputs = runtime.build_input_map(torch, spec, sample)

        self.assertEqual(tuple(inputs), ("a", "b"))

    def test_build_input_map_skips_nondifferentiable_integer_tensor_args(self) -> None:
        try:
            torch, linalg = runtime.import_generation_runtime()
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        op = next(row for row in linalg.op_db if row.name == "linalg.lu_solve")
        sample = next(iter(op.sample_inputs("cpu", torch.float64, requires_grad=True)))
        spec = SimpleNamespace(
            op="lu_solve",
            upstream_name="linalg.lu_solve",
            gradcheck_wrapper=None,
        )

        inputs = runtime.build_input_map(torch, spec, sample)

        self.assertEqual(tuple(inputs), ("a", "b"))


if __name__ == "__main__":
    unittest.main()
