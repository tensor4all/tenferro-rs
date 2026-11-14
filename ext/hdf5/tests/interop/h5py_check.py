#!/usr/bin/env python3
"""h5py side of the tenferro HDF5 adapter interop test.

Usage: h5py_check.py FILE

Mode 1 (read): the file has a `tensors` group, so this script asserts that the
Rust-written datasets follow the documented schema and hold the expected values.

Mode 2 (write): the `tensors` group exists but is empty, so this script writes
its own datasets (big-endian f64, h5py-default compound complex, bool enum) with
the same documented attributes for the Rust side to read back.
"""
import sys

import h5py
import numpy as np

SCHEMA_TYPE = "tenferro.Tensor"
SCHEMA_VERSION = 1


def expected(name, count):
    index = np.arange(count, dtype=np.float64)
    if name.startswith("f32"):
        return (index * 0.5 - 2.0).astype(np.float32)
    if name.startswith("f64"):
        return index * 0.5 - 2.0
    if name.startswith("i32"):
        return (index.astype(np.int64) * 7 - 11).astype(np.int32)
    if name.startswith("i64"):
        return index.astype(np.int64) * 7 - 11
    if name.startswith("c64"):
        return (index * 0.25 - 1j * index).astype(np.complex128)
    if name.startswith("bool"):
        return (index.astype(np.int64) % 3) == 1
    if name.startswith("rank0"):
        return np.float64(2.5)
    raise AssertionError(f"unexpected dataset {name}")


CASES = {
    "rank0_f64": ((), "f64"),
    "empty_f64": ((2, 0, 3), "f64"),
    "f32_3x4": ((3, 4), "f32"),
    "f64_3x4": ((3, 4), "f64"),
    "i32_2x2": ((2, 2), "i32"),
    "i64_2x2": ((2, 2), "i64"),
    "c64_2x2": ((2, 2), "c64"),
    "bool_5": ((5,), "bool"),
}

NUMPY_KIND = {
    "f32": "f4",
    "f64": "f8",
    "i32": "i4",
    "i64": "i8",
}


def check_schema(dataset, dtype_name):
    assert dataset.attrs["type"] == SCHEMA_TYPE, dataset.attrs["type"]
    assert int(dataset.attrs["version"]) == SCHEMA_VERSION
    assert dataset.attrs["dtype"] == dtype_name, dataset.attrs["dtype"]


def assert_rust_written(group):
    assert sorted(group) == sorted(CASES), sorted(group)
    for name, (shape, dtype_name) in CASES.items():
        dataset = group[name]
        check_schema(dataset, dtype_name)

        # HDF5 extents hold the reversed (C-order) dims of the column-major payload.
        assert dataset.shape == shape[::-1], (name, dataset.shape)

        if dtype_name in NUMPY_KIND:
            assert dataset.dtype.newbyteorder("=") == np.dtype(NUMPY_KIND[dtype_name])
            assert not dataset.dtype.byteorder == ">"
        elif dtype_name == "c64":
            # h5py stores complex as the compound {"r", "i"} and presents complex128.
            assert np.issubdtype(dataset.dtype, np.complexfloating), dataset.dtype
        elif dtype_name == "bool":
            # h5py recognises the int8 enum {FALSE, TRUE} as its own bool dtype.
            assert dataset.dtype == np.dtype(np.bool_), dataset.dtype

        if name == "empty_f64":
            assert dataset.size == 0
            continue
        # `.T` turns the stored C-order extents back into the logical tensor.
        logical = np.asarray(dataset).T
        want = expected(name, dataset.size).reshape(shape, order="F")
        if name == "rank0_f64":
            assert logical == want
        else:
            np.testing.assert_array_equal(logical, want, err_msg=name)


def write_from_python(group):
    # The dataset holds the transposed logical tensor, so its C-order payload is
    # the logical column-major payload and the extents are the reversed shape.
    f64 = (np.arange(6, dtype=np.float64) * 0.5 - 2.0).reshape((2, 3), order="F")
    dataset = group.create_dataset("py_be_f64", data=f64.T, dtype=">f8")
    write_schema(dataset, "f64")

    c64 = (np.arange(4, dtype=np.float64) * 0.25 - 1j * np.arange(4)).reshape(
        (2, 2), order="F"
    )
    dataset = group.create_dataset("py_c64", data=c64.T.astype(np.complex128), dtype=np.complex128)
    write_schema(dataset, "c64")

    bools = ((np.arange(5, dtype=np.int64) % 3) == 1).astype(np.bool_)
    dataset = group.create_dataset("py_bool", data=bools)
    write_schema(dataset, "bool")

    # Malformed fixtures for the Rust side's trust-boundary guards. Both are
    # written through the low-level API because the high-level one would repair
    # them: the enum dataset stores byte 2, which is no member of {FALSE, TRUE},
    # and the `type` attribute holds a byte that is not valid UTF-8.
    enum = h5py.h5t.enum_create(h5py.h5t.NATIVE_INT8)
    enum.enum_insert(b"FALSE", np.array(0, "i1"))
    enum.enum_insert(b"TRUE", np.array(1, "i1"))
    space = h5py.h5s.create_simple((2,))
    dsid = h5py.h5d.create(group.id, b"py_bad_bool", enum, space)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([2, 1], "i1"), mtype=h5py.h5t.NATIVE_INT8)
    dsid.close()
    enum.close()
    space.close()
    write_schema(group["py_bad_bool"], "bool")

    dataset = group.create_dataset("py_bad_utf8", data=np.zeros((2,), dtype=np.float64))
    write_schema(dataset, "f64")
    # A variable-length ASCII string, as the schema requires, holding a byte that
    # is not valid UTF-8: HDF5 never validates a string's contents.
    dataset.attrs.create(
        "type", np.array(b"\xff", dtype=object), dtype=h5py.special_dtype(vlen=bytes)
    )


def write_schema(dataset, dtype_name):
    dataset.attrs["type"] = SCHEMA_TYPE
    dataset.attrs["version"] = np.int64(SCHEMA_VERSION)
    dataset.attrs["dtype"] = dtype_name


def main():
    path = sys.argv[1]
    with h5py.File(path, "r+") as handle:
        group = handle["tensors"]
        if len(group) == 0:
            write_from_python(group)
        else:
            assert_rust_written(group)
    print("ok")


if __name__ == "__main__":
    main()
