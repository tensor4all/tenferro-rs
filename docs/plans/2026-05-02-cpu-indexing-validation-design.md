# CPU Indexing Validation Dispatch Design

**Date:** 2026-05-02

**Status:** Proposed dispatch spec

## Issues

Primary:

- #763: CPU reverse panics on I64 tensor, GPU accepts it
- #766: BUG: CPU reverse panics on I64 tensor, GPU accepts it
- #767: BUG: dispatch_tensor! / dispatch_binary! macros panic on I64 instead of returning Result
- #769: BUG: CPU gather/scatter/dynamic_slice use assert! for config validation instead of Result
- #779: BUG: index_tensor silently truncates float indices to int
- #788: BUG: typed_reverse uses assert! for axis bounds instead of Result
- #789: BUG: slice()/pad() public wrappers use .expect() and panic on error

Related if the same files are already being edited:

- #804: BUG: index_tensor panics on complex index tensors instead of returning error
- #814: BUG: assert!/panic! validation in CPU indexing/linalg/BLAS helpers

## Goal

Make CPU indexing helpers return normal `Result` errors for invalid user input
and unsupported index dtypes instead of panicking or silently coercing values.

The primary user-visible contract is:

- valid I64 reverse works on CPU, matching the CubeCL backend behavior,
- invalid indexing configurations return `Err`, not `panic!`,
- lossy or unsupported index dtype conversions return `Err`, not silent
  truncation or process panic.

## Scope

This dispatch is limited to CPU indexing behavior in
`tenferro-tensor/src/cpu/indexing.rs` and the smallest supporting changes
needed in tensor dispatch helpers.

It covers:

- `reverse` / `typed_reverse`,
- `gather`,
- `scatter`,
- `dynamic_slice`,
- public CPU convenience wrappers for `slice` and `pad`,
- index tensor conversion and validation.

It does not cover:

- GPU indexing performance (#765, #771),
- GPU I64 reduction policy (#764),
- broad `catch_backend_panic` redesign (#802),
- AD rules for indexing ops (#783, #787),
- dtype promotion semantics outside indexing (#811).

## Acceptance Specification

### I64 reverse

CPU reverse must support `Tensor::I64` when the input tensor is otherwise valid.
The implementation should mirror the explicit dtype dispatch style already used
by CPU slice, pad, and concatenate helpers.

### Error behavior

The following cases must return a repository error variant through `Result`:

- reverse axis out of bounds,
- invalid gather/scatter/dynamic_slice rank or config lengths,
- dynamic slice window sizes that exceed operand rank or bounds,
- complex index tensors,
- float index tensors containing non-finite values,
- float index tensors containing fractional values,
- float index tensors outside the exact integer range that can be represented
  by `i64`.

The implementation must not rely on `CpuBackend::catch_backend_panic` as the
normal error path.

### Public wrappers

If public convenience wrappers cannot return `Result` without an API change,
prefer one of these outcomes:

1. replace them with `try_*`-style public functions and update internal callers,
2. make non-Result wrappers private if no public contract depends on them,
3. stop and report the required API change before editing broadly.

Do not keep `.expect(...)` on user-controlled inputs.

### Dispatch macros

Do not mechanically put `Err(...)` inside macros whose expansion type is
currently `Tensor`. If macro behavior cannot be made Result-aware cleanly,
replace the indexing call sites with explicit dtype matches.

## Design

Use one small validation layer near each operation boundary. Each typed indexing
helper should receive already validated config where practical, but validation
may stay local when it depends on typed shape information.

Preferred shape:

1. top-level operation validates operation-level config and dtype support,
2. dtype dispatch calls a typed helper,
3. typed helper returns `Result<Tensor<T>>` or `Result<Tensor>`,
4. unsupported dtype paths build a normal `Error` with operation context.

For index tensors, introduce or reuse a helper that converts scalar index values
losslessly into `i64`. The helper must reject complex values and reject floats
that are not finite exact integers in `i64` range.

## Testing

Add focused regression tests in the owning crate. Keep tests outside production
modules unless the module already has tiny local tests.

Required cases:

- CPU reverse on `Tensor::I64` returns reversed data,
- CPU reverse with an out-of-bounds axis returns `Err`,
- gather/scatter/dynamic_slice invalid config returns `Err`,
- float index `1.5` returns `Err`,
- large float index outside exact integer range returns `Err`,
- complex index tensor returns `Err`,
- public slice/pad error paths no longer panic.

Run at least:

```bash
cargo test -p tenferro-tensor cpu::indexing
cargo fmt --all --check
```

If the exact test filter is not available, run the narrowest tenferro-tensor
test target that covers CPU indexing.

## Dispatch Prompt

```text
Implement the CPU indexing validation dispatch from
docs/plans/2026-05-02-cpu-indexing-validation-design.md.

Focus only on the listed CPU indexing issues. Do not redesign GPU indexing,
AD rules, dtype promotion, or catch_backend_panic. Replace panic/expect/assert
paths for user-controlled indexing inputs with normal Result errors, support
I64 reverse on CPU, and add focused regression tests. If a public non-Result
wrapper forces a breaking API decision, stop and report the narrowest required
change instead of broadening the patch.
```

## Review Checklist

- No new `panic!`, `assert!`, `assert_eq!`, `.unwrap()`, or `.expect()` on
  user-controlled indexing inputs.
- I64 reverse has the same CPU/GPU support decision.
- Float index conversion is explicit and lossless.
- Tests cover both successful I64 reverse and invalid-input errors.
- The patch does not touch unrelated GPU, AD, or linalg code.

## Stop Conditions

Stop and report if:

- making slice/pad wrappers non-panicking requires a public API break wider than
  CPU indexing,
- existing tests encode panic behavior as a public guarantee,
- fixing dispatch macros would require changing non-indexing call sites.
