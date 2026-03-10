import unittest

from generators import runtime, tolerance_audit


class HvpHelperTests(unittest.TestCase):
    def test_build_scalarized_observable_function_uses_real_part_for_complex_outputs(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        cotangent = {
            "value": torch.tensor([3.0 + 4.0j], dtype=torch.complex128)
        }

        def observable_fn(a):
            return (a,)

        scalarized = runtime.build_scalarized_observable_function(
            torch,
            observable_fn,
            output_names=("value",),
            cotangent=cotangent,
        )

        value = scalarized(torch.tensor([1.0 + 2.0j], dtype=torch.complex128))

        self.assertEqual(value.dtype, torch.float64)
        self.assertAlmostEqual(float(value.item()), 11.0)

    def test_compute_pytorch_hvp_matches_analytic_scalarized_hvp(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        inputs = {
            "a": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        }
        direction = {"a": torch.tensor([0.5], dtype=torch.float64)}
        cotangent = {"value": torch.tensor([1.5], dtype=torch.float64)}

        def observable_fn(a):
            return (a**3,)

        scalarized = runtime.build_scalarized_observable_function(
            torch,
            observable_fn,
            output_names=("value",),
            cotangent=cotangent,
        )

        pytorch_hvp = runtime.compute_pytorch_hvp(
            torch,
            scalarized,
            inputs=inputs,
            direction=direction,
        )

        self.assertTrue(torch.allclose(pytorch_hvp["a"], torch.tensor([9.0], dtype=torch.float64)))

    def test_compute_fd_hvp_matches_pytorch_hvp_for_scalarized_closure(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        inputs = {
            "a": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        }
        direction = {"a": torch.tensor([0.5], dtype=torch.float64)}
        cotangent = {"value": torch.tensor([1.5], dtype=torch.float64)}

        def observable_fn(a):
            return (a**3,)

        scalarized = runtime.build_scalarized_observable_function(
            torch,
            observable_fn,
            output_names=("value",),
            cotangent=cotangent,
        )

        pytorch_hvp = runtime.compute_pytorch_hvp(
            torch,
            scalarized,
            inputs=inputs,
            direction=direction,
        )
        fd_hvp = runtime.compute_fd_hvp(
            torch,
            scalarized,
            inputs=inputs,
            direction=direction,
            step=1e-6,
        )

        self.assertTrue(torch.allclose(pytorch_hvp["a"], fd_hvp["a"], rtol=1e-6, atol=1e-8))

    def test_hvp_residuals_measure_max_abs_and_rel_difference(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch runtime unavailable: {exc}")

        pytorch_hvp = {"a": torch.tensor([10.0, 5.0], dtype=torch.float64)}
        fd_hvp = {"a": torch.tensor([10.5, 4.5], dtype=torch.float64)}

        abs_residual, rel_residual = tolerance_audit.hvp_residuals(
            torch,
            pytorch_hvp,
            fd_hvp,
        )

        self.assertAlmostEqual(abs_residual, 0.5)
        self.assertAlmostEqual(rel_residual, 0.1)


if __name__ == "__main__":
    unittest.main()
