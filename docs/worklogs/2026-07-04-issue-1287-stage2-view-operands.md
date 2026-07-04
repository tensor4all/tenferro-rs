# Issue 1287 Stage 2: CUDA Dot-General Accumulation Over Borrowed Views

## Session Summary

Extended `CudaBackend::dot_general_read_into_accum` (stage 1, #1289) to accept
borrowed strided views (`TensorRead::View` / `TensorWrite::View`) over
GPU-resident device buffers on all three slots. A view contributes its own
shape / strides / element offset to the cuTENSOR descriptors; the effective
device pointer is `base + offset * size_of::<T>()`, and the descriptor
alignment is the alignment actually guaranteed by that byte address. Added the
missing public constructors `TypedTensor::backend_region_view` /
`backend_region_view_mut` in `tenferro-tensor` so callers can express matrix
regions of a flat device buffer at all (previously no public API could build a
mutable sub-region view over a backend buffer).

## Context Read

- Issue #1287 staged plan and the stage-1 work log
  (`2026-07-04-issue-1287-cuda-dot-accum.md`); stage-1 code in
  `crates/tenferro-gpu/src/cubecl/{mod.rs,gemm.rs}` and its ignored CUDA tests
  (`cubecl/tests/gemm_accum_tests.rs`).
- View data model in `crates/tenferro-tensor/src/types.rs`:
  `TypedTensorView{,Mut}` = `TensorBufferRef{,Mut}` (Host slice | shared
  `Arc<dyn BackendBuffer<T>>`) + `TensorLayout` (shape, isize strides, element
  offset) + `Placement`; `TensorLayout::from_parts` already proves the
  reachable span fits the backing buffer, and `validate_mutable_no_overlap`
  rejects self-aliasing mutable layouts.
- Existing view plumbing in `cubecl/dispatch.rs`
  (`cubecl_view_buffer`, `cubecl_view_mut_buffer`,
  `ensure_view{, _mut}_resident_on_runtime`) as the residency/downcast
  precedent, and `REPOSITORY_RULES.md` device-transfer, backend-buffer error,
  and negative-stride policy sections.

## Decisions Made

- **One generalized typed path**: `dot_general_typed_into_accum` now consumes
  `ReadOperand` / `WriteOperand` (owned compact tensor | borrowed view) and a
  per-operand `ResolvedOperand { ptr, strides, alignment }`. Owned operands
  keep the stage-1 behavior exactly (compact col-major strides, base pointer,
  256-byte descriptor alignment). The overwrite (non-accum) path is untouched.
- **Erased dispatch via variant accessors**: `CutensorScalar` gained
  `unwrap_view` / `unwrap_view_mut` / `unwrap_tensor_mut` (macro-implemented
  per dtype), so the `TensorRead`/`TensorWrite` enums dispatch through one
  generic `accum_erased::<T>` for f32/f64/c32/c64 instead of a 4x(2x2x2)
  match. Everything stays private to the `cubecl` module.
- **View constraints as explicit typed errors** (no silent fallback,
  validated before any degenerate-case early return):
  - host-backed views: explicit backend error via the existing
    `cubecl_view{_mut}_buffer` diagnostics (no hidden upload);
  - negative view strides: explicit error naming the cuTENSOR nonnegative
    stride contract (repo policy allows narrower adapter limits when stated
    at the boundary); zero strides on read views are passed through;
  - offset/extent bounds: defensive re-check that
    `offset + sum((dim-1)*stride)` fits the physical device buffer length,
    with overflow-checked arithmetic.
- **Per-descriptor alignment**: view descriptors pass the largest power of
  two dividing the effective byte address, capped at 256
  (`address_alignment`). Owned operands keep the stage-1 constant 256 rather
  than churning behavior.
- **New public region-view constructors** (`tenferro-tensor`):
  `TypedTensor::backend_region_view{,_mut}(shape, strides, offset)` build
  metadata-only views over the tensor's `Buffer::Backend` handle, reusing
  `TensorLayout::from_parts` bounds validation plus
  `validate_mutable_no_overlap` for the mutable variant; host buffers are
  rejected with explicit errors (host users keep the existing host
  constructors and `try_multi_slice_mut`). Backend buffers are `Arc`-shared,
  so distinct mutable regions can coexist; disjointness of regions used
  concurrently is the caller's contract, as with BLAS-style in-place APIs.
  This was required because no public API could construct a mutable
  sub-region view over a device buffer (host-only constructors, and
  `try_multi_slice_mut` returns `Ok(None)` for backend buffers).
- **Zero-sized contraction with a view output** (`out = beta * out`):
  `beta == 1` is a validated no-op; any other beta returns an explicit
  backend error because no strided in-place scale kernel exists yet. Owned
  outputs keep the stage-1 fill/scale kernels. This corner is unreachable for
  the intended matrix-region accumulation use.
- **Overlap semantics**: `C = D = out` unchanged; overlap between the out
  region and lhs/rhs regions is the caller's responsibility (BLAS
  convention), noted at the contraction call site.

## Rejected Alternatives

- Canonicalizing negative-stride or host views on the fly: hidden
  materialization/upload violates the device-transfer contract; explicit
  errors keep the caller in control.
- Computing descriptor alignment from the actual address for owned operands
  too: strictly more precise, but changes stage-1 descriptor inputs for no
  functional gain; left as-is to keep the stage-1 surface stable.
- Expressing the view-output `beta * out` degenerate case as a dummy
  cuTENSOR contraction or a fresh strided scale kernel: out of scope for the
  unreachable corner; documented limitation instead.
- Adding dtype-erased `Tensor::backend_region_view` wrappers: not needed by
  the backend path or tests; the typed constructors plus the existing
  `TensorView::F64(...)`-style wrapping suffice.

## Verification

- `cargo check -p tenferro-gpu` and `cargo check -p tenferro-gpu --features
  cuda` clean on the macOS dev host; `cargo test -p tenferro-gpu --features
  cuda --no-run` builds the CUDA test binaries.
- `cargo fmt --all` and CI-parity `cargo clippy --workspace --all-targets --
  -D warnings` clean; full `cargo test --workspace --release` green,
  including the new `backend_region_view` doctests and unit tests
  (layout/buffer sharing, host rejection, out-of-bounds span rejection,
  mutable aliasing rejection).
- New ignored CUDA tests in `cubecl/tests/gemm_accum_tests.rs`: offset
  strided regions on all three slots vs a host reference with untouched
  elements preserved bit-for-bit, two disjoint diagonal blocks of one flat
  buffer updated by successive calls, host-view operand rejection,
  negative-stride rejection, and the zero-contraction view-output beta
  contract. GPU execution of the ignored lane is pending on an A100 machine.

## Residual Risks

- The ignored CUDA lane has not run yet on real hardware (no CUDA on the dev
  host); cuTENSOR's acceptance of zero strides on read views and of
  sub-256-byte descriptor alignments is per documentation and must be
  confirmed on the A100 runner.
- The runtime out-of-bounds region check in `resolve_device_region` is
  defensive: public constructors already reject such layouts, so the error
  branch is not reachable through safe public API and is covered at the
  constructor level instead of by a CUDA runtime test.
- `out = beta * out` with `beta != 1` for view outputs remains an explicit
  backend error until a strided in-place scale kernel exists.
- WebGPU accumulation and further #1287 stages remain out of scope.
