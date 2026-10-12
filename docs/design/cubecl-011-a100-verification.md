# A100 verification of the CubeCL 0.11 migration

Status: first full run of the migrated branch's CUDA unit tests. 199 passed, 10
failed. The failures are classified below; none is infrastructure.

## Command

As documented in `AGENTS.md`, with the toolkit this host actually has
(`/usr/local/cuda` → CUDA 12.6, NVRTC 12.6.85; driver 580.126.09, A100 80GB
PCIe), build `-j 4`, tests 4 threads:

```bash
CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -j 4 -p tenferro-gpu --features cuda --lib -- --ignored --test-threads=4
```

Result: `199 passed; 10 failed; 0 ignored; 109 filtered out; finished in 51.94s`.
`--features cuda` builds clean (0 errors, 0 warnings) and the whole fusion test
set (12 tests) passes, including the multi-output and NaN-propagation cases.

## Failure classes

**1. Upstream `[bool]` launch argument panics (3 tests).**
`test_cubecl_integer_add_mul_compare_select_match_cpu`,
`test_cubecl_float_compare_select_and_clamp_match_cpu`,
`cuda_descriptor_supported_entries_match_cpu_smoke_cases`.

Backtrace (RUST_BACKTRACE=full) ends in upstream's own code, not tenferro's:

```
<bool as CubePrimitive>::__expand_size
<[bool] as LaunchArg>::register::{closure#0}
KernelLauncher::with_scope
```

panic at `cubecl-ir/src/interfaces/mod.rs:396`: "Expected type cube.bool to
implement dyn SizedType". `bool::__expand_as_type` returns `BoolType`
(`cubecl-core/src/frontend/element/bool.rs:45`), and `BoolType` is not
`SizedType` (`cubecl-ir/src/types/scalar.rs:355-361`), so any `&[bool]` kernel
parameter panics while the launcher registers it. Nothing in the migration is
involved: this is upstream 0.11's own bool-buffer path.

**2. Signed zero lost in `%` (2 tests).**
`test_scalar_div_rem_pow_match_cpu` (elementwise_tests.rs:1816) and
`test_cubecl_float_div_rem_preserve_ieee_special_values`:
`rem F32 index 5: actual=0.0 (0x00000000), expected=-0.0 (0x80000000)`, and the
same for the scalar-rhs case. Rust's `%` keeps the dividend's sign; the emitted
op does not. Candidate: the elementwise remainder now expands to `math.f_rem`
(`FRemOp`) where the 0.10 `Arithmetic::Rem` lowered with the sign, or the reverse
choice between `FRemOp` and `FModFloorOp`. Needs a look at the kernel's `rem`
helper and at CubeCL's own `%` lowering.

**3. `tests/mod.rs:61` panics (2 tests).**
`cuda_axpby_reads_an_offset_strided_source_in_place` and
`axpby_strided_source_kernel_compiles_once_per_rank`.

**4. Remaining numeric mismatches (3 tests).**
`test_log1p_small_x_f32_precision` (elementwise_tests.rs:1549),
`test_real_scalar_complex_binary_ops_match_cpu` (elementwise_tests.rs:204),
`scale_tensor_write_view_compact_prefix_preserves_outside` (scale.rs:310).

## Next

Diagnose 2-4 (each is a specific kernel or helper the migration touched),
then decide the bool-buffer question: either an upstream fix for `&[bool]`
launch arguments, or a tenferro representation change (bool tensors carried as
`u32`/`u8` buffers). The first is an upstream proposal and needs approval; the
second changes GPU memory representation, interop lengths and dtype mapping and
is therefore the user's call.
