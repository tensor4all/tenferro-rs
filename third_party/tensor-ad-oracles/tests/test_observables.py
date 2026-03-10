import unittest

from generators import observables


class FakeTensor:
    def __init__(self, label: str) -> None:
        self.label = label

    def abs(self):
        return FakeTensor(f"abs({self.label})")

    def __matmul__(self, other):
        return FakeTensor(f"({self.label}@{other.label})")

    def __eq__(self, other) -> bool:
        return isinstance(other, FakeTensor) and self.label == other.label

    def __repr__(self) -> str:
        return f"FakeTensor({self.label})"


class ObservableTests(unittest.TestCase):
    def test_identity_wraps_single_output(self) -> None:
        out = observables.apply_observable("identity", FakeTensor("x"))
        self.assertEqual(out, {"value": FakeTensor("x")})

    def test_identity_wraps_tuple_output(self) -> None:
        out = observables.apply_observable("identity", (FakeTensor("q"), FakeTensor("r")))
        self.assertEqual(out, {"output_0": FakeTensor("q"), "output_1": FakeTensor("r")})

    def test_identity_preserves_requested_tuple_keys_without_grad_filtering(self) -> None:
        out = observables.apply_observable(
            "identity",
            (FakeTensor("q"), FakeTensor("r")),
            preserve_identity_keys=("output_0", "output_1"),
        )
        self.assertEqual(out, {"output_0": FakeTensor("q"), "output_1": FakeTensor("r")})

    def test_identity_drops_nondifferentiable_integer_tuple_outputs(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        out = observables.apply_observable(
            "identity",
            (
                torch.ones(2, dtype=torch.float64),
                torch.ones(2, dtype=torch.int64),
            ),
        )
        self.assertEqual(tuple(out), ())

    def test_identity_drops_float_outputs_without_grad_path(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        a = torch.eye(3, dtype=torch.float64, requires_grad=True)
        out = observables.apply_observable("identity", torch.linalg.lu(a))

        self.assertLess(len(out), 3)

    def test_svd_u_abs_returns_abs_u(self) -> None:
        out = observables.apply_observable(
            "svd_u_abs",
            (FakeTensor("u"), FakeTensor("s"), FakeTensor("vh")),
        )
        self.assertEqual(out, {"u": FakeTensor("abs(u)")})

    def test_svd_observable_uniformizes_full_matrices(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        a = torch.randn(5, 3, dtype=torch.float64)
        result = torch.linalg.svd(a, full_matrices=True)

        out_u = observables.apply_observable("svd_u_abs", result)
        out_vh = observables.apply_observable("svd_vh_abs", result)
        out_uvh = observables.apply_observable("svd_uvh_product", result)

        self.assertEqual(tuple(out_u["u"].shape), (5, 3))
        self.assertEqual(tuple(out_vh["vh"].shape), (3, 3))
        self.assertEqual(tuple(out_uvh["uvh"].shape), (5, 3))

    def test_svd_vh_abs_keeps_s_and_abs_vh(self) -> None:
        out = observables.apply_observable(
            "svd_vh_abs",
            (FakeTensor("u"), FakeTensor("s"), FakeTensor("vh")),
        )
        self.assertEqual(
            out,
            {"s": FakeTensor("s"), "vh": FakeTensor("abs(vh)")},
        )

    def test_eigh_values_vectors_abs_keeps_values_and_abs_vectors(self) -> None:
        out = observables.apply_observable(
            "eigh_values_vectors_abs",
            (FakeTensor("values"), FakeTensor("vectors")),
        )
        self.assertEqual(
            out,
            {"values": FakeTensor("values"), "vectors": FakeTensor("abs(vectors)")},
        )

    def test_eig_values_vectors_abs_keeps_values_and_abs_vectors(self) -> None:
        out = observables.apply_observable(
            "eig_values_vectors_abs",
            (FakeTensor("values"), FakeTensor("vectors")),
        )
        self.assertEqual(
            out,
            {"values": FakeTensor("values"), "vectors": FakeTensor("abs(vectors)")},
        )
