# Uninitialized Full-Overwrite Provider Destinations (issue #1690)

Status: design (pre-implementation review target, revision 3).

## Problem

`dot_runtime.rs` canonical-operand allocation and `exec_session.rs` dot-output
allocation acquire **zeroed** pooled buffers, but both destinations are then
**fully overwritten** by the consumer (canonical-operand layout transform;
dot output with `beta == 0` via `Accum::Replace`, empty contractions write
zeros). The zero-fill is a wasted full pass (≈ 2× memory traffic on those
paths). Issue #1640 fixed the `structural.rs` sites; these two are the
deferred follow-up (#1690).

## Design (revision 3)

### 1. Uninitialized-destination traits (unsafe to implement) + structural witness

Two new **`unsafe trait`s** express the full-overwrite contract; implementing
requires an `unsafe impl` asserting it, so safe third-party code cannot
manufacture initialization proof:

```rust
unsafe trait CpuUninitLayoutTransformProvider: CpuLayoutTransformProvider {
    /// # Safety
    /// Must write every element of `output_bytes` before returning `Executed`;
    /// partial writes are UB at the caller's `assume_init`.
    unsafe fn materialize_into_uninit(
        &self,
        context: &CpuExecutionContext<'_>,
        input: &TensorRead<'_>,
        intent: CpuLayoutTransformIntent,
        conjugate: bool,
        output_bytes: &mut [MaybeUninit<u8>],
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;
}

unsafe trait CpuUninitGemmProvider: CpuGemmProvider {
    /// # Safety
    /// Must write every logical output element before `Executed`; invoked only
    /// for `beta == 0` (full-overwrite) accumulations.
    unsafe fn gemm_into_uninit(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmUninitRequest<'_, '_>,      // output-free prepared request
        output_bytes: &mut [MaybeUninit<u8>],
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;
}
```

**Enablement is a structural witness, not a boolean**: the base provider
traits gain one object-safe method whose default is `None`:

```rust
// on CpuLayoutTransformProvider and CpuGemmProvider respectively
fn uninit_provider(&self) -> Option<&dyn CpuUninitLayoutTransformProvider> { None }
fn uninit_provider(&self) -> Option<&dyn CpuUninitGemmProvider> { None }
```

Only a type that actually implements the `unsafe trait` (via `unsafe impl`)
can return `Some(self)` — a safe impl cannot construct a `&dyn
CpuUninit*Provider` it does not implement, so a `Some` witness is structural
proof the full-write contract was asserted. No boolean is used as proof.

`CpuGemmUninitRequest<'request, 'input>` is a public output-free prepared
request mirroring `CpuGemmRequest` (lhs/rhs/rows/columns/contracted/
batch_count/layouts/accumulation, no output — two lifetimes, since removing
`TensorWrite<'output>` removes the only use of `'output`), built by the
`DotGeneralRuntime` planner from the validated request. The GEMM seam's
uninit method sits beside the existing
`CpuGemmProvider::gemm(context, request: CpuGemmRequest)` and takes the same
`context: &CpuExecutionContext<'_>` (provider.rs:1187+, 1206-1210).

### 2. Caller pattern

At each site:

1. Resolve the executing provider (layout provider for the canonical operand;
   GEMM provider for the dot when `beta == 0`).
2. Get the witness: `provider.uninit_provider()`.
3. `Some(w)` → acquire `PooledUninitOutput`, call the `unsafe fn` in an
   `unsafe` block (precondition = the unsafe-impl contract), on `Executed`
   complete via `unsafe assume_init`.
4. `None` (provider opted out) → **go directly to the zeroed path** (no uninit
   checkout is acquired or discarded).
5. Opted-in but the method returns `Unsupported` → discard the uninit checkout
   (drop frees via `pool_discard_uninit`) and fall back to the zeroed path.
6. On `Err` → propagate the original error; never silently retry (an error may
   follow a partial write; matches current provider-error semantics).
7. `beta != 0` dot accumulations keep the zeroed path (the seed matters).

The discard-and-reallocate fallback fires only for opted-in providers that
return `Unsupported` for a given layout — never on the built-in hot path.

### 3. Safety argument

- Only `unsafe impl`s can appear as `Some` witnesses; a safe third-party
  provider is structurally unable to enable the uninit path.
- The destination travels only as `&mut [MaybeUninit<u8>]` until the `unsafe
  assume_init` handoff, whose precondition is exactly the unsafe-impl
  full-write guarantee. No `&mut [T]`/`TensorWrite` is fabricated over
  uninitialized storage.
- Built-in full-write guarantees are kernel-verified: layout transform
  traverses the destination; faer `Accum::Replace` for beta zero; empty
  contractions write zeros.

## Bench evidence (expected)

Before/after (pinned, release): mid-size dot (128×128 f64, 1 thread) where
the operand/output zero-fills are measurable, and a tiny 2×2 case. Report the
memset share removed honestly (a few % mid-size; negligible tiny/large). Add
a behavioral test: an opted-out provider still gets the zeroed path; an
opted-in provider returning `Unsupported` falls back correctly.

## Non-goals

- No change to the initialized-output provider methods or their callers.
- No change to `beta != 0` (accumulate) dot paths.
- No new public Tensor/backend API beyond the two `unsafe trait`s, the
  witness methods (default `None`), and `CpuGemmUninitRequest`; existing
  third-party providers are unaffected (default opt-out).
