# HDF5 payload adapter for tensors (#2611)

## Decisions

- One optional crate, `ext/hdf5` (`tenferro-hdf5`, `publish = false`), as a
  standalone excluded manifest next to `ext/sparse` and `ext/tropical`. It keeps
  HDF5 out of `tenferro-tensor`'s dependency graph, and keeps the root workspace
  test and coverage lanes free of a runtime `libhdf5` requirement (this crate's
  tests and doctests open real files). A workspace member was rejected for that
  second reason; `fmt` is still covered by `scripts/ci/run_profile.py`, and a
  dedicated `hdf5-adapter` CI job covers clippy and tests.
- One backend: `hdf5-rt` at a pinned revision, re-exported as
  `tenferro_hdf5::hdf5` so callers always hold the handles the adapter expects.
  A second link-time backend (`hdf5-metno`) behind a mutually exclusive feature
  was rejected: it doubles the backend surface and the CI matrix for no v1 user.
- Schema v1 is one dataset per tensor with `type`/`version`/`dtype` attributes
  and the **reversed** HDF5 extents. HDF5 extents are C-order and tenferro
  storage is column-major, so the reversed extents make the stored payload
  identical to the column-major buffer: both directions are copy-free, and
  `np.asarray(ds).T` is the logical tensor in h5py. Logical extents plus an
  "order" attribute would mis-index in every generic reader, and a transposing
  write costs a full copy.
- Complex is the h5py compound `{"r", "i"}` (byte-identical to
  `num_complex::Complex<T>`), bool is the h5py int8 enum `{FALSE, TRUE}`.
  HDF5 2.x's native `H5T_COMPLEX` is not read in v1 because `hdf5-rt` cannot yet
  decode it; TRIQS trailing-dimension-2 complex is rejected explicitly.
- No overwrite policy: a taken link name fails with `AlreadyExists`, and
  replacing means unlinking first, because the caller owns the group. A
  temporary-link atomic-replace state machine was rejected as unreachable scope.
  Flushing belongs to the `File` owner; HDF5 has no crash atomicity.
- A typed read validates attributes, dataspace, shape/byte count, dtype, static
  rank and the HDF5 datatype before it allocates, then allocates the payload
  exactly once. `ReadOptions::max_elements` bounds untrusted files. Backend
  storage is rejected rather than transferred or materialized, and a
  non-compact view is rejected rather than compacted.
- `bool` is read through an `i8` buffer whose bytes are validated to be 0 or 1
  and then reinterpreted in place, so no unvalidated byte ever becomes a `bool`
  and the read still costs one allocation.
- Attributes are validated as untrusted input before the binding reads them: each
  must be one scalar element (a null dataspace also has rank zero, and reading a
  scalar out of it would take an uninitialized value), `version` must have an
  integer datatype (`read_scalar` would otherwise convert `1.5` to `1`), and text
  bytes are checked with `str::from_utf8` because the binding's `as_str()` is
  `from_utf8_unchecked` and HDF5 never validates string contents.
- The dtype-erased read loads the dataset and its metadata once and shares the
  payload path with the typed read, instead of re-opening the dataset per dtype.

## Verification conclusions and constraints

- Round trips cover `f32 f64 c32 c64 i32 i64 bool` across rank 0, rank 1-4, a
  zero-sized axis, static-rank reads, `Dynamic` (plain and promoted) inputs and
  a compact offset view; float and complex bit patterns (NaN payloads, `-0.0`,
  subnormals, infinities) are preserved.
- Negative coverage: `External` dtype, backend-resident storage, non-compact
  view, rank > 32, existing link, invalid names, typed dtype and static rank
  mismatch, missing/foreign/unknown attributes, null-dataspace attributes,
  non-integer `version`, unsupported version, datatype disagreeing with the
  `dtype` attribute, null dataspace, overflowing shape, element limit, and a
  write into a read-only file leaving no link.
- Allocation counting proves the write path allocates no payload-sized buffer
  (per dtype threshold, `bool` included) and a read allocates exactly one,
  owned by the returned tensor; the measurement sees Rust allocations on the
  measuring thread, so it supports the no-adapter-staging claim, not a claim
  about libhdf5's internal buffers.
- h5py interoperability is asserted in both directions with the documented
  schema, including a big-endian `f64` dataset written by h5py. The h5py fixture
  also writes the two malformed cases the Rust guards reject: an int8 enum
  payload holding byte 2 (`InvalidBool`) and a variable-length ASCII `type`
  attribute holding a non-UTF-8 byte (`SchemaViolation::AttributeNotUtf8`).
- Constraints: `hdf5-rt` is a git dependency, so the crate cannot be published
  until it is on crates.io; native `H5T_COMPLEX` files and HDF5.jl's `re`/`im`
  complex and `B8` bool encodings are rejected in v1.
- The adapter was verified against libhdf5 2.0.0 loaded at run time; the CI job
  installs the distribution's serial HDF5 (1.10 on ubuntu-24.04) instead, so the
  hosted run is the 1.x evidence.
- A tensor with zero elements gets a one-element chunk shape from `hdf5-rt`
  (HDF5 cannot allocate a contiguous layout with no elements); readers are
  unaffected and non-empty tensors stay in one contiguous chunk, which the guide
  states explicitly.

## Review record

- Plan review, before any code: `ollama-cloud/glm-5.3-flash` (read-only
  subagent): `VERDICT: APPROVE` with one IMPORTANT finding (write the payload from
  `TypedTensorView::as_slice()`, not `host_storage()`) and four minor ones, all
  applied to the plan.
- Post-implementation review of the finished diff: `openai/gpt-6.1-sol`
  (read-only, `scripts/pi_review.sh`): changes requested — two blockers (null
  dataspace and non-UTF-8 attribute bytes reaching unsafe binding paths), three
  important (bool allocation threshold, `version` datatype conversion, untested
  `InvalidBool`) and two minor (dtype-erased payload not asserted, false
  no-chunking claim). All were reproduced against the sources and fixed; the
  dtype-erased double metadata read and the codec hint in `Error::UnsupportedDType`
  were removed as the review's over-engineering pass suggested.
