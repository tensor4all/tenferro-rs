# Optional XLA Backend Design

Date: 2026-06-14
Status: Draft for review
Related issue: https://github.com/tensor4all/tenferro-rs/issues/984

## Context

tenferro already has a backend-independent traced graph path:

```text
TracedTensor -> GraphCompiler -> GraphProgram -> ExecProgram
```

Native execution then runs the `ExecProgram` through `GraphExecutor<B>` and a
`TensorBackend`. That path must remain the primary execution model for dynamic
shape workloads, extension runtime dispatch, CPU/CUDA/WebGPU backend work, and
the existing cache and device-transfer contracts.

Issue #984 asks for an optional XLA path for static-shaped graph execution. The
purpose is not to replace the native backend. The purpose is to use
StableHLO/PJRT as:

- a performance reference for static-shaped graphs,
- an optional execution backend where the dependency cost is acceptable, and
- a forcing function that keeps tenferro's graph IR compiler-friendly.

StableHLO should be treated as an external portability boundary, not as a new
in-process replacement for `ExecProgram`.

## Goals

- Add an experimental, feature-gated XLA path for static-shaped traced graphs.
- Keep XLA dependencies out of `tenferro-runtime` and out of the native
  `TensorBackend` trait.
- Lower from `GraphProgram` / `ExecProgram` into StableHLO in a separate
  `tenferro-xla` crate.
- Verify generated StableHLO by compiling and executing it through PJRT in
  Phase 2, not only by snapshot-testing text.
- Start with a deliberately small op set that covers simple elementwise graphs
  and einsum-lowered contraction graphs.
- Return explicit unsupported errors for unsupported ops, dtypes, dynamic
  extents, extension families, layouts, or PJRT capabilities.
- Keep executable caching owned by the XLA executor with bounded defaults,
  clear/configure APIs, and stats.

## Non-Goals

- Do not replace native CPU, CUDA, or WebGPU execution.
- Do not extend `TensorBackend` to model whole-program compilation.
- Do not support dynamic shapes in the first XLA implementation.
- Do not support arbitrary physical strides as external PJRT buffers in the
  first implementation.
- Do not lower linalg, SVD truncation, `DynamicTruncate`, `PadToMatch`, or
  StableHLO `custom_call` in the first milestones.
- Do not silently fall back from XLA execution to native CPU or GPU execution
  when an op is unsupported.

## Architecture

Add a new workspace crate:

```text
crates/tenferro-xla
```

The crate owns the optional XLA path:

```text
GraphProgram / ExecProgram
  |
  v
tenferro_xla::lowering
  |
  v
StableHLO module text or bytecode
  |
  v
tenferro_xla::pjrt
  |
  v
PJRT executable
```

`tenferro-runtime` should expose only the narrow owner-scoped accessors needed
for external lowering. If the existing `GraphProgram` / `ExecProgram` fields
are too private, add small read-only public or `doc(hidden)` accessors rather
than making execution internals broadly public.

The native path remains unchanged:

```text
GraphProgram -> GraphExecutor<B: TensorBackend> -> native backend
```

The XLA path is a peer executor, not a `TensorBackend` implementation:

```rust
pub struct XlaExecutor { ... }

impl XlaExecutor {
    pub fn lower(&self, program: &GraphProgram) -> Result<StableHloModule>;
    pub fn compile(&mut self, program: &GraphProgram) -> Result<XlaExecutable>;
    pub fn run_to_host(
        &mut self,
        program: &GraphProgram,
        inputs: &[XlaInput<'_>],
    ) -> Result<Vec<Tensor>>;
}
```

Exact names can change during implementation, but the ownership boundary should
not: `XlaExecutor` owns XLA/PJRT state and caches.

## Phase 1: StableHLO Lowering

Phase 1 builds the lowering layer and validates its structure without requiring
PJRT to be installed.

In scope:

- Create `tenferro-xla` with no default dependency on PJRT libraries.
- Add a `stablehlo` lowering module that emits StableHLO MLIR text.
- Lower only static-shaped programs.
- Add a capability table that maps supported `ExecOp` variants to lowering
  functions.
- Add tests that lower small graphs and assert the StableHLO structure.
- Add unsupported tests for dynamic extents, unsupported ops, extension ops,
  unsupported dtypes, and multi-output cases not yet handled.

Initial op set:

- `Constant`
- `Add`
- `Multiply`
- `Negate`
- `Convert`
- `Reshape`
- `BroadcastInDim`
- `Transpose`
- `ReduceSum`
- `DotGeneral`

Deferred until the initial op set is compiling and executing through PJRT:

- `Subtract` lowering as `Add + Negate` or as a direct StableHLO subtraction
  once the binary elementwise helper is settled.
- `ReduceProd`, `ReduceMax`, and `ReduceMin` through the same reducer-region
  helper used by `ReduceSum`.

Phase 1 must not treat generated text snapshots as proof of correctness. It
only proves that the lowering boundary, capability checks, and module structure
are deterministic and reviewable.

## Phase 2: PJRT Execution Verification

Phase 2 adds actual compilation and execution through PJRT so generated
StableHLO is checked by a real compiler/runtime.

In scope:

- Add a `pjrt` feature to `tenferro-xla`.
- Load a PJRT C API plugin through an explicit path or environment variable.
- Compile the Phase 1 StableHLO module with `PJRT_Program.format = "mlir"`.
- Upload host inputs to PJRT buffers.
- Execute the compiled program.
- Download outputs to tenferro `Tensor` values.
- Compare outputs against native `GraphExecutor<CpuBackend>` for the same
  `GraphProgram`.
- Separate compile latency from steady-state execution latency in benchmarks.

Validation tiers:

- CPU PJRT tests are the default Phase 2 correctness target when a CPU PJRT
  plugin is available.
- GPU PJRT tests are environment-gated and ignored by default. On this local
  development machine, they should run against the available NVIDIA A100 GPU.
- CI must not require a GPU unless a dedicated GPU runner is configured.

Phase 2 success means the same static-shaped program can run through both:

```text
GraphExecutor<CpuBackend>
XlaExecutor -> StableHLO -> PJRT
```

and produce equal shapes, dtypes, and values within dtype-appropriate
tolerances.

## Layout Boundary

tenferro runtime tensors use compact column-major storage. XLA/PJRT host buffer
behavior and default layouts may not match that physical order.

The first implementation should make the boundary explicit:

- Before sending a host tensor to PJRT, convert it to the physical order
  expected by the emitted StableHLO/PJRT layout policy.
- After downloading a PJRT output, convert it back to a tenferro column-major
  `Tensor`.
- Keep the conversion code inside `tenferro-xla`; do not hide it in
  `tenferro-runtime` or native backends.
- Document the conversion as an XLA boundary cost and keep it out of
  steady-state kernel timing when benchmarking the compiled executable.

This is an explicit external-backend ABI boundary, not a hidden native
CPU/GPU transfer. Later milestones may add layout attributes or PJRT layout
control to reduce or remove these conversions.

## Cache Ownership

`XlaExecutor` owns compiled executable caches.

The executable cache key should include at least:

- a structural fingerprint of the `ExecProgram`,
- input dtypes and concrete shapes,
- output dtypes and concrete shapes,
- lowering version,
- StableHLO compatibility/version data when available,
- PJRT plugin identity,
- PJRT platform and topology fingerprint when available,
- compiler options,
- layout policy,
- enabled XLA feature profile.

Cache rules:

- Default cache capacity is bounded.
- Users can resize and clear the cache.
- Users can inspect entry count and retained byte estimates.
- Compile-time and runtime cache stats are exposed separately when both exist.
- Unsupported programs are not cached as successful executables.

## Capability And Error Model

`tenferro-xla` should use an explicit capability table rather than scattered
matches in one large lowering function.

Each supported op lowering declares:

- supported `ExecOp` variant,
- supported dtypes,
- rank/static-shape requirements,
- required StableHLO op or region shape,
- whether the op can produce multiple outputs,
- whether the op needs special layout handling.

Errors should name the failing instruction index and op:

```text
xla lowering unsupported op at instruction 7: DynamicTruncate requires dynamic shape support
```

Unsupported cases must fail before PJRT compile whenever they are known from
tenferro metadata. PJRT errors should be wrapped with the plugin/platform name
and the phase (`compile`, `host upload`, `execute`, `host download`).

## Extension Operations

Phase 1 and Phase 2 should reject `ExecOp::Extension` unless the extension has
already been expanded into core `ExecOp` instructions before XLA lowering.

Future extension support should use a separate XLA lowering registry, not the
native `ExtensionRuntime` registry. Extension crates may participate by:

- expanding into core operations at graph-build or compile time, or
- registering an XLA lowering that emits core StableHLO-compatible operations.

Custom calls are deferred. When custom calls are introduced, they should be
owned by an explicit XLA backend profile and capability catalog, not hard-coded
inside extension crates.

## Tests

Phase 1 tests:

- Lower `x + y`, `x * scalar`, `neg(x)`, `reshape`, `transpose`, and
  `broadcast_in_dim`.
- Lower `reduce_sum` with one axis and multiple axes.
- Lower a simple matrix multiplication to `dot_general`.
- Lower a small contraction graph after existing einsum expansion exposes
  primitive ops.
- Verify deterministic module text for stable snapshots.
- Verify unsupported dynamic extents and unsupported ops fail with explicit
  diagnostics.

Phase 2 tests:

- Execute the Phase 1 examples through PJRT and compare against
  `GraphExecutor<CpuBackend>`.
- Cover `f32` and `f64` first.
- Cover bool only when compare/select enters the supported op set.
- Use max absolute or relative residuals in assertion failures.
- Keep GPU PJRT tests ignored or environment-gated.

Benchmark tests:

- Separate lowering time, PJRT compile time, first run, and steady-state run.
- Keep host repack/upload/download costs visible as separate measurements.
- Compare against native CPU execution for the same `GraphProgram`.

## Documentation

Update developer-facing docs when implementation begins:

- `docs/spec/backend-contract.md`: record `tenferro-xla` as an optional
  peer executor over `ExecProgram`, not a `TensorBackend`.
- `docs/design/` or `docs/internals/`: add an XLA backend design page once the
  first implementation lands.
- README and user-facing docs should mention the XLA path only as experimental
  and static-shape-only until PJRT execution is working.

Public rustdoc for `tenferro-xla` must include runnable examples for APIs that
do not require an installed PJRT plugin. PJRT examples may be feature-gated and
document the required environment variables.

## Risks

- PJRT distribution and plugin loading may be the hardest part, especially for
  reproducible local and CI setup.
- StableHLO text generation without a verifier can drift. Phase 2 mitigates
  this by compiling and executing the generated module.
- Column-major tenferro layout may add boundary repacking cost. The first
  implementation should measure it explicitly instead of hiding it.
- A too-large first op set would make unsupported behavior ambiguous. Keep the
  first milestones narrow.
- Making `ExecProgram` internals broadly public would freeze internal runtime
  details prematurely. Prefer narrow read-only accessors.

## Success Criteria

Phase 1 is complete when:

- `tenferro-xla` lowers the initial static op set to deterministic StableHLO
  module text.
- Unsupported programs fail with explicit diagnostics.
- No XLA/PJRT dependency is pulled into `tenferro-runtime`.
- Native execution tests remain unchanged and passing.

Phase 2 is complete when:

- The same lowered modules compile and execute through PJRT.
- PJRT outputs match native CPU outputs for the initial op set.
- Compile time and steady-state execution time can be measured separately.
- GPU PJRT verification can run on the local A100 environment when the required
  plugin is available, while ordinary CI remains GPU-independent.
