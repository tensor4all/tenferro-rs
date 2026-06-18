# Context Engine Refactor Design

## Context

GitHub issue #876 asks for a breaking traced-execution API cleanup. The
current public API exposes `Engine<B: TensorBackend>` as the value passed to
`TracedTensor::eval`. In practice, `Engine<B>` owns several different concerns:

- the backend execution target
- the backend runtime cache
- the graph compile cache
- einsum parse and contraction-plan caches
- the execution slot workspace

That makes `TracedTensor` hard to explain: graph values are backend-independent,
but their public evaluation API requires a backend-dependent `Engine<B>`.
`Engine`, `Backend`, and `EagerContext` also overlap conceptually in docs.

This change intentionally does not preserve the old public API. The repository
rules prefer removing confusing public surfaces instead of keeping compatibility
shims.

## Goals

- Make `TracedTensor` clearly lazy, backend-independent, and non-executing.
- Replace public `Engine<B>` with separate compilation and execution objects.
- Make cache ownership match responsibility boundaries.
- Make tensor import/export memory order explicit.
- Rename eager runtime concepts so names describe ownership and behavior.
- Update user docs around data order, execution modes, and backend/device
  choices.

## Non-Goals

- Preserve `Engine<B>` or `.eval(&mut engine)` compatibility.
- Hide compilation and execution behind a replacement convenience wrapper.
- Add `ndarray` integration.
- Rewrite the lower execution IR beyond what the API split requires.

## Public API Shape

Traced execution becomes explicit:

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let x = TracedTensor::from_vec_row_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let y = (&x * &x).reduce_sum(&[0]);

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&y)?;

let mut executor = GraphExecutor::new(CpuBackend::new());
let out = executor.run(&program)?;
assert_eq!(out.shape(), &[3]);
```

For placeholders, compilation accepts input specs and execution accepts concrete
bindings:

```rust
let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
let y = &x + &x;

let mut compiler = GraphCompiler::new();
let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])?;

let input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let mut executor = GraphExecutor::new(CpuBackend::new());
let out = executor.run_with_inputs(&program, &[(&x, &input)])?;
```

The exact input-binding API can be adjusted during implementation, but the
boundary remains fixed: `GraphCompiler` validates graph/input specs and produces
a portable program; `GraphExecutor<B>` receives a program plus concrete inputs
and runs it on backend `B`.

## Responsibilities

| Type | Backend-dependent? | Responsibility |
|---|---:|---|
| `TracedTensor` | No | Lazy graph value, metadata, graph-building methods, AD graph transforms |
| `GraphCompiler` | No | Resolve/materialize/compile traced graphs, own backend-independent caches |
| `GraphProgram` | No | Portable compiled execution plan plus input metadata needed for execution |
| `GraphExecutor<B>` | Yes | Own backend `B`, backend runtime cache, slot workspace, execution caches |
| `EagerRuntime` | Yes | Own eager backend state and scalar-loss gradient storage |

`GraphProgram` should initially wrap the existing `ExecProgram` instead of
forcing a deep IR rename. This keeps the public API clean while preserving the
current optimized execution path.

## Cache Ownership

Move from one mixed owner:

```rust
Engine<B> {
    backend,
    backend_cache,
    compile_cache,
    einsum_cache,
    einsum_parse_cache,
    slot_workspace,
}
```

to:

```rust
GraphCompiler {
    compile_cache,
    einsum_parse_cache,
    static_einsum_plan_cache,
}

GraphExecutor<B: TensorBackend> {
    backend,
    backend_cache,
    einsum_plan_cache,
    slot_workspace,
}
```

The compile cache is keyed by graph structure and input specs, not backend
state. The einsum parse cache belongs with compilation because it is notation
parsing, not execution. Static concrete-shape einsum decomposition also stays
compiler-side because it rewrites the graph before execution. Runtime
`NaryEinsum` planning for symbolic shapes remains executor-side because runtime
shapes and backend-specific cost models may affect planning.

Both compiler and executor expose bounded cache controls, clear methods, and
stats. CPU-specific buffer-pool controls move from `Engine<CpuBackend>` to
`GraphExecutor<CpuBackend>`.

## Traced Evaluation

Remove `TracedTensor::eval`, `TracedTensor::eval_with_inputs`, and public
`eval_all(engine, ...)` APIs. Replace them with compiler/executor methods:

- `GraphCompiler::compile(&TracedTensor) -> Result<GraphProgram>`
- `GraphCompiler::compile_many(&[&TracedTensor]) -> Result<GraphProgram>`
- `GraphCompiler::compile_with_input_specs(...) -> Result<GraphProgram>`
- `GraphExecutor::run(&GraphProgram) -> Result<Tensor>`
- `GraphExecutor::run_many(&GraphProgram) -> Result<Vec<Tensor>>`
- `GraphExecutor::run_with_inputs(...) -> Result<Tensor>`

During implementation, helper code that currently lives in `TracedTensor`
(`compile_with_inputs`, `compile_with_input_specs`, input resolution, deferred
zero tangent handling) should move behind compiler-owned private functions.
`TracedTensor` should retain AD transform methods (`grad`, `vjp`, `jvp`, HVP
composition) because those build new graph values and do not execute.

`checkpoint` currently evaluates through `Engine`. It should be rewritten to
use `GraphCompiler` and `GraphExecutor` explicitly, or redesigned as a caller
operation that materializes a tensor before replacing graph data. It must not
reintroduce hidden execution through `TracedTensor`.

## Memory Order API

Current `from_vec(shape, data)` means column-major but the name does not say
that. Replace it with explicit constructors:

```rust
TypedTensor::from_vec_col_major(shape, data)
TypedTensor::from_vec_row_major(shape, data)

Tensor::from_vec_col_major(shape, data)
Tensor::from_vec_row_major(shape, data)

TracedTensor::from_vec_col_major(shape, data)
TracedTensor::from_vec_row_major(shape, data)
```

Storage remains contiguous column-major internally. Row-major constructors
convert into the internal column-major buffer at construction time.

Add explicit exports:

```rust
typed.into_vec_col_major()
typed.into_vec_row_major()

tensor.into_vec_col_major::<T>()
tensor.into_vec_row_major::<T>()
```

If a borrowed export is added, name it `to_vec_*` or `try_to_vec_*` so ownership
is clear. Existing `as_slice` remains a typed host-buffer borrow and must be
documented as physical column-major order.

Remove or make crate-private the ambiguous public `from_vec` and
`try_into_vec` APIs. Update all examples and public docs to avoid them.

## Eager Runtime Rename

Rename `EagerContext` to `EagerRuntime`. It owns backend state and gradient
slots; it is not a passive context.

Public eager shape becomes:

```rust
let runtime = EagerRuntime::new();
let x = runtime.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
let loss = (&x * &x).reduce_sum(&[0])?;
loss.backward()?;
```

Update module re-exports so `tenferro::EagerRuntime` and
`tenferro::eager_tensor::EagerRuntime` are the public names. Do not keep
`EagerContext` as a public alias unless a compile-only compatibility alias is
explicitly required later.

## Documentation Plan

Rewrite user docs around these concepts:

1. Data model: shape, dtype, column-major physical storage, row-major import
   and export helpers.
2. Execution modes:
   - concrete immediate execution: `Tensor` / `TypedTensor` + backend
   - eager scalar-loss AD: `EagerTensor` + `EagerRuntime`
   - traced compile/execute: `TracedTensor` + `GraphCompiler` + `GraphExecutor`
3. Backends/devices:
   - CPU backend
   - CUDA backend
   - explicit upload/download boundaries
4. API reference and examples.

Avoid presenting the system primarily as "four tensor layers"; that hides the
more important distinction between data, graph construction, compilation, and
execution.

Docs to update include:

- `README.md`
- `tenferro/README.md`
- `tenferro/src/lib.rs` crate docs
- `docs/getting-started/**`
- `docs/guides/choosing-an-api.md`
- `docs/guides/eager-operations.md`
- `docs/guides/memory-order.md`
- `docs/guides/devices-and-gpu.md`
- `docs/architecture/tenferro-crates.md`

Do not update historical files under `docs/plans/` except this design and the
follow-up implementation plan.

## Testing

Add or update integration tests because the `tenferro` facade crate has
`[lib] test = false`.

Required coverage:

- `GraphCompiler` can compile a traced graph without owning any backend.
- `GraphExecutor<CpuBackend>` runs compiled single-output and multi-output
  programs.
- Placeholder input specs are validated during compilation.
- Concrete placeholder bindings are validated during execution.
- Compiler cache stats and bounds are independent of executor/backend caches.
- Compiler-side static einsum plan stats are independent of executor-side
  runtime einsum plan stats.
- Executor cache stats, backend runtime cache stats, slot workspace reuse, and
  CPU buffer-pool controls live on `GraphExecutor`.
- Repeated symbolic einsum execution reuses executor-side contraction plans.
- `Tensor`, `TypedTensor`, and `TracedTensor` row-major constructors produce the
  same logical values as column-major constructors.
- Explicit row-major and column-major exports return the requested order.
- Public docs and checked examples use `GraphCompiler`, `GraphExecutor`,
  `EagerRuntime`, and explicit memory-order constructors.

Run focused checks during development, then the repository PR checklist before
creating a PR.

## Risks

- This touches a large public surface, so mechanical updates can hide real
  behavior changes. Use focused integration tests before sweeping call-site
  rewrites.
- Input binding currently mixes compilation and concrete values. The refactor
  must keep symbolic-shape and deferred-zero tangent behavior intact.
- CUDA traced tests may require feature-gated verification on a configured
  machine. CPU behavior should be fully checked locally.
- Removing ambiguous constructors will touch many tests. Prefer mechanical
  replacements with explicit row/column intent over changing expected values.
