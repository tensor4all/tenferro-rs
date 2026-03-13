import unittest

from validators import encoding


class ValidatorEncodingTests(unittest.TestCase):
    def test_decode_real_tensor_round_trips_shape_and_dtype(self) -> None:
        encoded = {
            "dtype": "float64",
            "shape": [2, 2],
            "order": "row_major",
            "data": [1.0, 2.0, 3.0, 4.0],
        }

        tensor = encoding.decode_tensor(encoded)

        self.assertEqual(str(tensor.dtype), "torch.float64")
        self.assertEqual(tuple(tensor.shape), (2, 2))
        self.assertEqual(tensor.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_decode_complex_tensor_round_trips_pairs(self) -> None:
        encoded = {
            "dtype": "complex128",
            "shape": [2],
            "order": "row_major",
            "data": [[1.0, 0.5], [-2.0, 3.0]],
        }

        tensor = encoding.decode_tensor(encoded)

        self.assertEqual(str(tensor.dtype), "torch.complex128")
        self.assertEqual(tuple(tensor.shape), (2,))
        self.assertEqual(tensor.tolist(), [complex(1.0, 0.5), complex(-2.0, 3.0)])

    def test_decode_float32_tensor_round_trips_shape_and_dtype(self) -> None:
        encoded = {
            "dtype": "float32",
            "shape": [2],
            "order": "row_major",
            "data": [1.25, -3.5],
        }

        tensor = encoding.decode_tensor(encoded)

        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertEqual(tuple(tensor.shape), (2,))
        self.assertEqual(tensor.tolist(), [1.25, -3.5])

    def test_decode_complex64_tensor_round_trips_pairs(self) -> None:
        encoded = {
            "dtype": "complex64",
            "shape": [1],
            "order": "row_major",
            "data": [[1.0, -0.5]],
        }

        tensor = encoding.decode_tensor(encoded)

        self.assertEqual(str(tensor.dtype), "torch.complex64")
        self.assertEqual(tuple(tensor.shape), (1,))
        self.assertEqual(tensor.tolist(), [complex(1.0, -0.5)])
