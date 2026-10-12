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

Result of the latest run: `201 passed; 8 failed; 0 ignored; 109 filtered out;
finished in 52.71s` (the first run was 199/10).
`--features cuda` builds clean (0 errors, 0 warnings) and the whole fusion test
set (12 tests) passes, including the multi-output and NaN-propagation cases.

## Failure classes

**0. Fixed since the first run: the float remainder's signed zero (2 tests).**
`test_scalar_div_rem_pow_match_cpu` and
`test_cubecl_float_div_rem_preserve_ieee_special_values` used the textbook
remainder formula with `if remainder == 0 { x * 0.0 }` to carry the dividend's
sign. 0.11's `math.f_mul` constant fold turns `x * 0.0` into `+0.0`, losing it;
a three-lane probe kernel showed `x*0 = 0x00000000` for `x = -4.0`. The kernels
now use `%` (`math.f_rem` → `fmod`). See
[cubecl-011-ieee-constant-fold.md](./cubecl-011-ieee-constant-fold.md).

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

**2. Upstream `log1p` precision regression (1 test).**
`test_log1p_small_x_f32_precision`: `log1p(1e-7)` returned `1.1920928e-7`
(rel_err 0.19) where `1e-7f32.ln_1p()` is `9.9999994e-8`. That is exactly
`ln(1 + x)` computed in f32, because 0.11's C++ backend lowers `Log1pOp` through
the polyfill `(input + 1).ln()` (`cubecl-core/src/frontend/polyfills.rs:207`,
used at `cubecl-cpp/src/shared/unary.rs:321`) instead of the target's native
`log1pf`. The fork's 0.10 backend called
`D::compile_instruction_log1p_scalar(f, input)`, i.e. native, so this is a
regression the migration to upstream introduces. `recip` (`1/x`) is polyfilled
the same way but is IEEE-exact; `erf` stays native except on Metal.
Tenferro's test is the precision requirement; a tenferro-local libm
replacement would be exactly the local workaround the migration rules forbid,
so this waits on an upstream fix.

**3. Signed zero in the real-scalar complex path (1 test).**
`test_real_scalar_complex_binary_ops_match_cpu`: `C32 scalar+complex[2].im:
actual=-0.0, expected=0.0`. Same family as the remainder defect — a zero's sign
depends on operand order and on the arithmetic identities the IR applies.

**4. `scale` writes outside the compact prefix (1 test).**
`scale_tensor_write_view_compact_prefix_preserves_outside`: the first three
elements match, the rest come back 4x the expected values, i.e. the elements
outside the prefix were scaled as well.

Root cause: 0.11 gives a buffer's device-visible length the **handle's** byte
size, not the length the caller passed. `KernelLauncher::process_buffer`
computes `buffer_len = array.handle.size_in_used() / elem_size` and ignores
`BufferBinding.length` (`cubecl-core/src/compute/launcher.rs:202`), and the
frontend's `length` field is only read by `BufferBinding::into_tensor()`
(`cubecl-core/src/frontend/container/slice/launch.rs:118-133`). So
`BufferArg::from_raw_parts(handle, 3)` over a 6-element allocation still shows
`out.len() == 6` in the kernel, and the flat scale kernel correctly scales all
six.

Trimming the handle to the region in `CubeclPreparedAccess::into_array_arg` was
tried and **ruled out by measurement**: it takes the 201/8 result to 129/80,
because tenferro's strided bindings address the allocation relative to its
start (kernel shape/stride arguments, `BufferArg::from_raw_parts` with the
allocation handle) and need the handle to keep covering it. The viable fix is
targeted: the flat kernels that bound themselves with `.len()` (scale,
fill_zero) must receive the region length as an explicit kernel argument, or
resolve the region through a binding that carries it, rather than changing what
every binding means.

This is a contract gap worth reporting with the other upstream items: the safety
comment on `from_raw_parts` ("Specifying the wrong length may lead to
out-of-bounds reads and writes") implies the length governs the region, while the
launcher ignores it in favour of the handle's size.

**5. `download_tensor` errors in the axpby strided path (2 tests).**

Not a transfer bug: the download's claim check surfaces a real launch failure
from earlier in the test, which the 0.11 `get_resource` claim reports ("The bytes
were never written ... A launch error happened"). The launch failed because the
generated CUDA does not compile:

```
default_program(111): error: no instance of overloaded function "fma" matches the argument list
            argument types are: (const cuDoubleComplex, const cuDoubleComplex, const cuDoubleComplex)
  cuDoubleComplex const v223 = fma(v49, v56, v69);
```

Cause: `cubecl-opt/src/passes/inst_combine.rs` turns `a*b + c` into an `FmaOp`
(`lone_product`/`as_fma`, lines 28-47) without restricting it to real float
scalars, and the C++ backend lowers `FmaOp` to `fma(...)`, which has no complex
overload. Tenferro's complex axpby kernel forms exactly that pattern
(`y + alpha * x` with a single use of the product). Upstream fix: keep complex
out of the FMA formation.

## Upstream items found by this run

All five are things the fork had solved, or upstream defects that the migration
exposes. Each needs an upstream change (`tracel-ai/*`), i.e. the user's approval.

1. IEEE-unsound constant folds in the math dialect — three tenferro sites, one
   fixed locally, one still failing. [cubecl-011-ieee-constant-fold.md](./cubecl-011-ieee-constant-fold.md)
2. `&[bool]` launch arguments panic (`BoolType` is not `SizedType`) — 3 tests.
3. `Log1pOp` loses precision in the C++ backend where the fork called the native
   `log1pf` — 1 test.
4. `BufferArg`'s length is ignored by the launcher in favour of the handle's
   byte size — 1 test; the local handle-trim fix was measured and ruled out.
5. `inst_combine` forms `FmaOp` on complex operands — 2 tests.

## Next

Diagnose 3-5 (tenferro-side), then decide the two upstream items: the
bool-buffer question (upstream `SizedType` for `BoolType`, or a tenferro
representation change to `u8`/`u32` buffers) and the `log1p` lowering (upstream
C++ native `log1pf`, as the fork had). Both need the user's approval as
`tracel-ai/*` work.
