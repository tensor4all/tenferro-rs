# HDF5 payload I/O

`tenferro-hdf5` (`ext/hdf5`) stores ordinary tenferro tensor values in HDF5
datasets: `TypedTensor<T, R, Host>`, the same tensor once it became `Dynamic`
while still host-resident, and the dtype-erased `Tensor`.

It is an **optional adapter crate**. No tenferro compute crate depends on it, it
is not a workspace member of the root build, and it adds no HDF5 edge to
`tenferro-tensor`. Add it by path or git revision:

```toml
[dependencies]
tenferro-hdf5 = { git = "https://github.com/tensor4all/tenferro-rs", rev = "<rev>" }
```

The crate is `publish = false`: `hdf5-rt` is a git dependency, so it cannot be
published to crates.io as it stands.

## Backend

One backend: [hdf5-rt](https://github.com/tensor4all/hdf5-rt), re-exported as
`tenferro_hdf5::hdf5`. Callers open files and groups through that re-export, so
their handles are always the ones the adapter expects.

`hdf5-rt` loads `libhdf5` at **run time** and needs nothing at build time. Install
it on Linux with `apt-get install libhdf5-dev`, or use a conda-forge `hdf5`. The
loader tries the bare `libhdf5.so` and `libhdf5_serial.so` first (so
`LD_LIBRARY_PATH` and `ld.so.conf` are honoured), then the usual serial-install
paths. If nothing loads, the library panics inside the first HDF5 call; the
adapter never initializes HDF5 itself.

## I/O ownership

The caller owns the `File` and the `Group`, and the adapter touches exactly one
link name inside the group you pass:

```rust,no_run
use tenferro_hdf5::{hdf5, read_typed, write_typed};
use tenferro_tensor::{DynRank, Host, TypedTensor};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let file = hdf5::File::create("tensors.h5")?;
let group = file.create_group("tensors")?;

let tensor: TypedTensor<f64, DynRank, Host> =
    TypedTensor::from_host_vec_col_major(vec![2, 3], (0..6).map(f64::from).collect())?;
write_typed(&group, "a", tensor.as_view())?;

let back: TypedTensor<f64, DynRank, Host> = read_typed(&group, "a")?;
assert_eq!(back.as_slice(), tensor.as_slice());
# Ok(())
# }
```

Each call takes a group and a single link name. There is no path syntax, no
archive mode and no hidden policy.

- **No hidden transfer.** Backend-resident storage is rejected
  (`Error::NotHostResident`). Nothing is downloaded, uploaded, synchronized or
  materialized; transfer and canonicalization are yours to make explicit.
- **No hidden copies.** A compact column-major view is written straight from its
  host buffer, and a read allocates exactly one payload buffer, which the
  returned tensor adopts. A non-compact view is rejected (`Error::NotCompact`).
- **No numeric conversion.** dtype, shape and values round-trip unchanged, bit
  patterns included.
- **No overwrite.** A taken link name fails with `Error::AlreadyExists`. To
  replace a payload, `group.unlink(name)` first — the group is yours.

## Schema, version 1

One dataset per tensor, with three scalar attributes:

| attribute | written as | value |
|---|---|---|
| `type` | variable-length UTF-8 string | `"tenferro.Tensor"` |
| `version` | `i64` | `1` |
| `dtype` | variable-length UTF-8 string | `f32`, `f64`, `c32`, `c64`, `i32`, `i64`, `bool` |

The HDF5 dataspace holds the shape **reversed**: logical column-major
`[d0, d1, ..., dn-1]` is stored as extents `[dn-1, ..., d1, d0]`. HDF5 extents are
C-order (last dimension fastest) while tenferro storage is column-major, so the
reversed extents make the stored payload identical to the column-major buffer and
both directions are copy-free. In a generic HDF5 reader the logical tensor is
therefore the transpose:

```python
import h5py
import numpy as np

with h5py.File("tensors.h5", "r") as handle:
    dataset = handle["tensors/a"]
    assert dataset.attrs["type"] == "tenferro.Tensor"
    assert dataset.attrs["version"] == 1
    logical = np.asarray(dataset).T          # column-major logical tensor
```

Rank 0 becomes a scalar dataspace, and a zero-sized axis becomes a zero extent
with no payload write. The layout is contiguous and v1 writes no filters or
compression. A non-empty tensor is stored in one contiguous chunk; a tensor with
zero elements has nothing to store, so `hdf5-rt` gives it a one-element chunk
shape instead. Readers do not care either way, but a tool that inspects the
layout will see chunking on empty tensors only.

| dtype | HDF5 datatype | h5py reads it as |
|---|---|---|
| `f32`, `f64` | `H5T_IEEE_F32LE`, `F64LE` | `float32`, `float64` |
| `i32`, `i64` | `H5T_STD_I32LE`, `I64LE` | `int32`, `int64` |
| `c32`, `c64` | compound `{"r": F @0, "i": F @sizeof(F)}` | `complex64`, `complex128` |
| `bool` | int8 enum `{FALSE=0, TRUE=1}` | `bool` |

Complex is the h5py compound convention, which is also what `tensor4all-hdf5`
writes: `num_complex::Complex<T>` is byte-identical to that compound, so complex
is copy-free as well. `bool` is the h5py enum convention, the same one TRIQS
writes.

Known interoperability limits, all rejected with a typed error rather than
silently mis-read:

- HDF5 2.x's native `H5T_COMPLEX` class is **not** read in v1; `hdf5-rt` cannot
  yet decode it.
- HDF5.jl writes complex as a `re`/`im` compound and `bool` as an
  `H5T_NATIVE_B8` bitfield, so its complex and bool payloads are rejected;
  numeric dtypes interoperate in both directions.
- A TRIQS/ALPS trailing-dimension-of-2 complex encoding carries a `__complex__`
  attribute and is rejected.
- Attributes must be variable-length strings, as written above; a fixed-length
  `type` or `dtype` attribute is rejected.
- `DType::External` payloads are rejected (`Error::UnsupportedDType`); a Rust
  `TypeId` is never persisted, so an external scalar set needs an explicit codec
  that v1 does not have.

## Validation before allocation

A read validates everything it can before allocating: the `type`/`version`/`dtype`
attributes, the dataspace (a null dataspace is rejected), the shape product and
byte count against `isize::MAX`, the requested dtype and static rank, and the
HDF5 datatype against the declared dtype (byte order is ignored, so a
big-endian file converts and reads normally). Only then does it allocate the
payload once.

Attributes are treated as untrusted input of their own: each must be one scalar
element, `version` must be an integer datatype (`1.5` is not silently converted
to `1`), and text attributes must hold valid UTF-8 — HDF5 checks a string's
declared character set, never its bytes.

`ReadOptions::max_elements(n)` caps the element count for files you do not trust:

```rust,no_run
use tenferro_hdf5::{read_typed_with, ReadOptions};
# use tenferro_hdf5::hdf5;
# use tenferro_tensor::{DynRank, Host, TypedTensor};
# fn run(group: &hdf5::Group) -> Result<(), tenferro_hdf5::Error> {
let options = ReadOptions::default().max_elements(1 << 20);
let tensor: TypedTensor<f64, DynRank, Host> = read_typed_with(group, "a", &options)?;
# let _ = tensor;
# Ok(())
# }
```

## Errors

Failures are typed: `Error::DTypeMismatch`, `RankMismatch`, `NotCompact`,
`NotHostResident`, `UnsupportedDType`, `RankTooLarge`, `InvalidName`,
`AlreadyExists`, `UnsupportedVersion`, `ElementLimit`, `ShapeOverflow`,
`Schema { violation }` (missing/foreign/unknown attributes, datatype
disagreement, null dataspace), `InvalidBool`, `InvalidAttributeValue`, `Hdf5`
(any underlying HDF5 call) and `Tensor` (a tensor constructor rejection).

## Testing this adapter

Its tests need a runtime `libhdf5`, so they are not part of the root workspace
test run. The dedicated `hdf5-adapter` CI job installs `libhdf5-dev` and runs:

```bash
cargo fmt --manifest-path ext/hdf5/Cargo.toml --all --check
cargo clippy --manifest-path ext/hdf5/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path ext/hdf5/Cargo.toml
TENFERRO_HDF5_INTEROP=1 cargo test --manifest-path ext/hdf5/Cargo.toml interop
```

The h5py interoperability test uses `python3` with `h5py` (override with
`TENFERRO_HDF5_PYTHON`).
