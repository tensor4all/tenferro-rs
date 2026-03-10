import unittest


from generators import encoding


class FakeDType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class FakeTensor:
    def __init__(self, data, *, dtype: str) -> None:
        self._data = data
        self.dtype = FakeDType(dtype)
        self.shape = self._infer_shape(data)

    def _infer_shape(self, value):
        if isinstance(value, list):
            if not value:
                return (0,)
            return (len(value),) + self._infer_shape(value[0])
        return ()

    def is_complex(self) -> bool:
        return "complex" in str(self.dtype)

    def detach(self):
        return self

    def clone(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._data


class EncodingTests(unittest.TestCase):
    def test_encode_real_tensor_uses_row_major_shape_and_data(self) -> None:
        tensor = FakeTensor([[1.0, 2.0], [3.0, 4.0]], dtype="float64")

        encoded = encoding.encode_tensor(tensor)

        self.assertEqual(encoded["dtype"], "float64")
        self.assertEqual(encoded["shape"], [2, 2])
        self.assertEqual(encoded["order"], "row_major")
        self.assertEqual(encoded["data"], [1.0, 2.0, 3.0, 4.0])

    def test_encode_complex_tensor_emits_real_imag_pairs(self) -> None:
        tensor = FakeTensor([[1.0 + 2.0j, 3.0 - 4.0j]], dtype="complex128")

        encoded = encoding.encode_tensor(tensor)

        self.assertEqual(encoded["dtype"], "complex128")
        self.assertEqual(encoded["shape"], [1, 2])
        self.assertEqual(encoded["data"], [[1.0, 2.0], [3.0, -4.0]])

    def test_encode_scalar_tensor_uses_empty_shape(self) -> None:
        tensor = FakeTensor(2.5, dtype="float64")

        encoded = encoding.encode_tensor(tensor)

        self.assertEqual(encoded["shape"], [])
        self.assertEqual(encoded["data"], [2.5])

    def test_encode_tensor_normalizes_torch_prefixed_dtype(self) -> None:
        tensor = FakeTensor([[1.0]], dtype="torch.float64")

        encoded = encoding.encode_tensor(tensor)

        self.assertEqual(encoded["dtype"], "float64")

    def test_encode_tensor_materializes_unallocated_torch_storage(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        a = torch.eye(3, dtype=torch.float64, requires_grad=True)
        da = torch.ones_like(a)

        def fn(x):
            return torch.linalg.slogdet(x)

        _, jvp = torch.func.jvp(fn, (a,), (da,))

        encoded = encoding.encode_tensor(jvp[0])

        self.assertEqual(encoded["dtype"], "float64")


if __name__ == "__main__":
    unittest.main()
