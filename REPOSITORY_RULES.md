# Repository Rules

These are tenferro-specific rules. Apply them on top of the shared tensor4all
rules from `tensor4all-agent-rules`.

## Public Surface Drift

- `README`, rustdoc, and examples must not claim capabilities beyond the current public surface.
- When the public API changes, check for stale names, stale capability claims, and deleted paths in `README`, rustdoc, and examples before considering the work complete.

## Naming Style

- Prefer `tensor4all` over `Tensor4all` in project prose, documentation,
  issue text, comments, and contributor notes unless quoting an external name
  or preserving an existing proper noun.

## Public Surface Discipline

- Keep the public API intentionally small. Prefer `pub(crate)` for types,
  functions, traits, modules, fields, and helper constructors unless external
  users are expected to call them directly.
- Do not make implementation details public just because another module needs
  them. First consider whether the code belongs in the same crate/module, or
  whether a narrower crate-private helper is the right abstraction.
- Public APIs should be selected deliberately and documented as user-facing
  contracts. If an item is primarily for tests, benchmarks, internal planning,
  execution dispatch, lowering, caching, or backend glue, it should normally be
  private or `pub(crate)`.
- Before adding or keeping a `pub` item, ask whether it is useful outside this
  repository and whether tenferro is prepared to support its semantics as a
  public contract. If the answer is unclear, keep it `pub(crate)` and expose a
  smaller high-level API instead.
- Tensor operation names are public vocabulary. Use unsuffixed operation names
  for owned compact tensor inputs, and add a `_read` suffix only for APIs that
  explicitly accept borrowed views or `TensorRead`-style input references.
- Metadata-only layout/view operations must use a `_view` suffix, for example
  `transpose_view`, `slice_view`, or `reshape_view`. Do not use `_view` for
  operations that allocate, canonicalize, execute kernels, or transfer data.

## External Contribution Intake

- Bug-fix PRs may be reviewed as merge candidates when they fix behavior that
  is already intended by current docs, specs, or tests.
- New features must start as feature request issues. Do not treat a
  new-feature implementation PR as the source of truth unless maintainers
  already accepted the linked issue and agreed that implementation should
  start.
- New-feature PRs opened before an accepted issue may be closed and redirected
  to an issue. Prototype code should be linked from the issue as reference
  material.
- Accepted feature issues, specs, and repository rules are the source of truth
  for implementation. External prototype code may inform implementation only
  when its license and provenance are compatible with this repository.
- When maintainer or AI-assisted implementation is based on external prototype
  code, preserve appropriate copyright notices, license obligations,
  attribution, and links to the original prototype or issue discussion.

## Standard Extension Boundary

- Standard operation families (`tenferro-einsum`, `tenferro-linalg`,
  `tenferro-fft`, and future peers) are first-class crates, not modules of a
  broad `tenferro` facade.
- The workspace intentionally has no root `tenferro` facade crate. Do not add
  operation-family facade paths such as `tenferro::einsum`,
  `tenferro::linalg`, or `tenferro::fft`; users import operation crates
  directly.
- Users import operation crates directly and register runtimes explicitly, for
  example `tenferro_einsum::einsum` plus
  `executor.register_extension(tenferro_einsum::register_runtime)`.
- Optional capabilities are feature boundaries, not new operation-family
  crates. Put operation-specific AD support behind an `autodiff` feature in the
  owning operation crate instead of adding `*-ad` companion crates.
- User-facing operation crates expose backend features as `cuda` and `rocm`.
  Do not document or require a public `gpu` feature on those crates; use
  internal `cfg(any(feature = "cuda", feature = "rocm"))` checks or internal
  helper features when needed.
- Extension crates depend on the runtime, tensor, AD, or GPU crate they need;
  dependency flow must not require a facade crate to depend back on them.
- Extension AD rules should be owned by an explicit `tenferro_ad::AdContext`
  or rule set. Process-global registration may remain only as a compatibility
  bridge, not as the primary design for new APIs.

## Oracle Gate

- Do not add or keep an AD rule implementation in the mainline without a
  corresponding oracle family.
- Prefer oracle families with both Torch reference data and finite-difference checks.
- If a Torch reference is not available, a finite-difference-only oracle is acceptable.
- If no corresponding oracle exists yet, add it to `tensor-ad-oracles` before treating the rule as a supported mainline AD rule.

## Rule Source Of Truth

- `PrimitiveOp::linearize` and `PrimitiveOp::transpose_rule` (in `tenferro-internal-ops/src/ad/`)
  are the semantic source of truth for AD rules.
- These are graph-level rules that emit ops into a `FragmentBuilder`.
  `tidu::differentiate` calls `linearize`; `tidu::transpose` calls `transpose_rule`.
- The canonical tenferro AD model is graph-level `linearize` plus
  `transpose_rule`. Do not model tensor primitive AD by implementing
  `chainrules_core::ReverseRule<StdTensorOp>` or `ForwardRule<StdTensorOp>`;
  those traits are value/tape-level interfaces and are not the standard
  primitive-op rule surface in this repository.
- Avoid introducing ChainRules-style `frule`/`rrule` terminology for new
  tenferro AD APIs. Use `linearize` for JVP graph emission and
  `transpose_rule` for transposed linear graph emission. A future
  externally supplied pullback API must be designed separately instead of
  being mixed into the current primitive rule contract.
- Reverse-mode support is not always a direct `transpose_rule` arm on the
  primal op. Some ops are supported by first applying `linearize` and then
  transposing the emitted linear primitive graph. Before filing or closing an
  AD support issue, check the machine-readable AD support manifest in
  `tenferro-internal-ops/src/ad/support.rs`; `SupportedViaLinearize` means a missing
  direct transpose arm is intentional.
- Reference JAX's implementations (`jax/_src/lax/lax.py`, `jax/_src/lax/linalg.py`)
  when implementing new AD rules.

## AD Rule Coverage

- Every `linearize` / `transpose_rule` implementation must have a corresponding
  finite-difference integration test that verifies numerical correctness.
- For linalg ops, prefer oracle families with both Torch reference data and
  finite-difference checks when available in `third_party/tensor-ad-oracles/`.

## No Ad Hoc Fixes

- Do not add ad hoc fixes that violate DRY, KISS, or layering.
- Do not introduce compatibility shims, duplicated logic, or downstream reach-through into lower layers when the correct fix belongs in an existing seam or high-level API.

## File Organization

Keep source files small and focused, but do not split files solely to reduce
line count. Treat ~1000 lines as a soft review trigger, not as a mechanical
limit. A file that remains one coherent concern may stay above that size until
there is a clear behavior, abstraction, feature, or ownership boundary to
extract.

When splitting a large file, the split must make the code easier to reason
about. Prefer boundaries such as parsing, plan computation, execution,
dispatch, public API, AD rules, operation families, backend glue, validation,
or cache ownership. Avoid arbitrary `part1` / `part2` splits and avoid moving
code into tiny files that force readers to chase one concept across many
modules.

Use line count to decide where to inspect first. Use responsibility, change
frequency, public/private API boundaries, and human navigation to decide
whether and how to split.

## Unit Test Organization

For Rust modules, keep production source files focused on production code.
Do not keep inline `#[cfg(test)]` blocks in normal modules unless the file is a
genuinely tiny leaf module and the test is trivially small. Prefer
module-local test directories such as `src/<module>/tests/*.rs` and leave only
`#[cfg(test)] mod tests;` in the source file. Reserve crate-root `tests/` for
integration tests. Do not use `include!` to inject test files into modules.

When splitting tests, optimize for keeping AI and human reading context clean:
a developer reading `src/**` should not need to scroll through large unit-test
blocks to understand the implementation. Prefer splitting larger extracted test
suites by concern rather than keeping one monolithic test module.

Tests follow implementation ownership.

- Public facade crates should prefer integration tests for user-visible
  behavior.
- Private implementation details must be tested in the crate that owns the
  implementation, typically an internal crate, not through a public facade
  crate.
- If a crate sets `[lib] test = false`, do not add `src/**/tests`, inline
  `#[cfg(test)] mod tests`, or other crate-local unit-test entrypoints to that
  crate.
- If a private helper in a facade crate needs direct unit testing, move that
  helper into the owning internal crate instead of re-enabling facade-crate lib
  tests.
- This rule is enforced by repository contract tests and must stay green in CI.

## Performance And Layout Rules

### Complexity Budget

- Do not introduce accidental `O(n^2)` behavior in graph construction,
  metadata propagation, key hashing/equality, compilation, or execution
  scheduling. If a quadratic algorithm is intentional, document why the input
  size is bounded or why the tradeoff is acceptable.
- Avoid repeatedly cloning, hashing, formatting, or scanning whole graph
  histories, metadata scope lists, tensor input maps, or structural keys inside
  per-node/per-op loops. Prefer stable IDs, interning, cached fingerprints with
  exact equality checks, or persistent/shared data structures.
- When optimizing compiler or graph-build overhead, measure scaling across
  increasing graph sizes, not only one fixed benchmark case.

### Materialization And Copies

- Production tensor paths must not silently materialize dense temporary buffers
  whose memory or time scales with an unconstrained product of tensor
  dimensions. Dense/reference implementations are allowed only when the API is
  explicitly named and documented as dense, reference, or debug behavior.
- Avoid dense copy-in/copy-out around operations that can consume strided
  views, borrowed slices, backend-native buffers, or metadata-only layout
  changes.
- Do not introduce hidden CPU-GPU transfers. Follow the public device-transfer
  policy: callers explicitly upload/download tensors, while execution pipeline
  internals may move constants or scalar metadata only where the backend
  contract documents that behavior.
- When a copy or materialization is required by an output contiguity contract,
  backend limitation, or external ABI boundary, make that boundary explicit in
  the implementation and cover it with tests.

### Device Transfer And Backend Buffer Errors

- tenferro follows the PyTorch convention: no implicit CPU-GPU transfer.
  Tensors must already live on the device required by the backend operation.
- Canonicalization is allowed only within the existing placement. Host views may
  copy into host compact tensors; GPU backend views may canonicalize or copy
  back on the GPU. These boundaries must not download or upload tensor payloads
  unless the API is explicitly a transfer API.
- User code must explicitly upload CPU tensors before CUDA backend execution
  and explicitly download CUDA tensors before CPU-only execution or host value
  inspection.
- A GPU backend op receiving a CPU tensor must return
  `Error::BackendFailure` with a diagnostic that says the op expected a GPU
  tensor and points users to `upload_tensor()`.
- A `Result`-returning CPU backend op receiving a backend/GPU buffer must
  return `Error::BackendFailure` where the buffer is detected at the CPU
  backend boundary. The diagnostic should say to download the tensor to host
  before CPU execution.
- Direct host-inspection APIs such as `TypedTensor::host_data()` and
  `TypedTensor::host_data_mut()` may panic with a diagnostic on backend
  buffers because their signatures return slices, not `Result`.
- Execution pipeline internals may handle placement for documented cases:
  `Constant` ops may auto-upload through `upload_host_tensor()`, and
  host-dependent ops such as `ShapeOf` or `DynamicTruncate` may read metadata
  or download single scalar values as documented by the backend contract.
- Unsupported CUDA op or dtype combinations must return an explicit error, not
  silently fall back to CPU execution.

### Dense Layout And Linear Algebra

- tenferro uses column-major (Fortran order) dense storage: the leftmost
  dimension has the smallest stride and varies fastest in memory.
- Owned runtime tensors remain compact column-major only. Arbitrary strides,
  offsets, transposes, slices, and reverse views live on
  `TypedTensorView`/`TypedTensorViewMut` or metadata-only layout values until an
  explicit same-placement canonicalization boundary is reached.
- Public flat-buffer constructors, exports, examples, FFI contracts, and docs
  must state or preserve column-major semantics.
- Do not add row-major compatibility shims or hidden row-major round-trips in
  library code. If external data is naturally row-shaped, convert privately at
  that explicit boundary.
- For batched GEMM-style layouts, put contraction/compute dimensions on the
  left and batch dimensions on the right. In column-major storage, this keeps
  each batch slice contiguous for the GEMM kernel.
- Use existing tensor/backend abstractions for GEMM, einsum, and dense linear
  algebra. Do not reimplement linalg kernels locally in downstream layers when
  the CPU/GPU backend already owns the operation.

### Range Checks And Slicing

- Public indexing and slicing APIs must validate rank, bounds, steps, output
  shape, and empty/singleton boundary behavior at the API boundary or planning
  boundary.
- Tensor layout metadata and runtime typed views may use signed strides and
  negative slice steps when reachable-range validation proves every logical
  element maps inside the backing allocation. Zero step remains invalid. Do not
  reject negative strides solely because they are negative. Narrower adapter
  APIs may document stricter compatibility limits, but those limits must be
  explicit at the API boundary.
- After validation, hot loops and kernels should not repeat the same range
  checks per element. Carry validated shape/stride/offset metadata into the
  inner implementation instead.
- Prefer safe Rust patterns that help LLVM eliminate bounds checks: iterate
  over slices directly, slice once before the loop, use `chunks_exact` when the
  chunk size divides the length, and add explicit pre-loop assertions for
  validated index ranges. Use unchecked indexing only as a last resort.
- Prefer metadata-only slices, strided views, or backend-native slice kernels
  over dense copies. If a slice must allocate, document and test the reason.
- Slice, reshape, transpose, gather, scatter, pad, concatenate, and reverse
  implementations must preserve column-major shape/stride/offset semantics.
- If unchecked indexing or unsafe pointer access is used after validation, keep
  the validation invariant close to the unsafe block and add tests for full
  range, empty or singleton slices, lower and upper boundaries, out-of-range
  errors, rank mismatch, and non-contiguous slices.

### CPU Kernel Implementation

- No naive CPU loop fallbacks. CPU tensor kernels must use optimized
  implementations unless the operation is explicitly listed as an exception
  below.
- Required CPU implementations by operation category:

  | Category | Required implementation |
  |---|---|
  | Elementwise (`add`, `mul`, `neg`, `exp`, ...) | `strided-kernel` (`map_into`, `zip_map2_into`, etc.) |
  | Reduction (`reduce_sum`, `reduce_prod`, ...) | `strided-kernel` (`reduce`, `reduce_axis`) |
  | Structural (`transpose`, `broadcast`, `extract_diag`) | `strided-kernel` (`permute` + `copy_into`, `broadcast`, `diagonal_view`) |
  | GEMM (`dot_general`) | faer (`cpu-faer`) or BLAS (`cpu-blas`) |
  | Linalg (`svd`, `qr`, `cholesky`, `eigh`, `solve`) | faer (`cpu-faer`) or LAPACK (`cpu-blas`) |

- Exceptions with dedicated implementations are `reshape` (metadata-only),
  `embed_diagonal`, and indexing ops such as gather, scatter, slice, pad,
  concatenate, and reverse.
- CPU provider features are additive. At least one of `cpu-faer` or `cpu-blas`
  must be enabled; enabling both is valid and must compile. `CpuBackend` owns
  the runtime provider selection. `CpuBackend::new()` selects the default
  compiled provider (`cpu-faer` when present), and explicit constructors or
  application configuration select BLAS/LAPACK when that provider is compiled
  and linked.

### Tensor Core Data Model

- `tenferro-tensor-core` owns backend-independent host tensor metadata and
  contiguous host storage: `DType`, `TensorScalar`, `HostTensor<T>`,
  dynamic `Tensor`, host/dynamic views, `TensorRef`, `ShapeVec`, `StrideVec`,
  `SliceSpec`, and metadata-only `reshape_view`, `transpose_view`, and
  `slice_view`.
- `tenferro-tensor-core` must not depend on CUDA, GPU backends, backend buffers,
  provider selection, or execution backend traits. Its owned `Tensor` must not
  grow inherent `TensorBackend` execution helpers.
- Core views and layouts must validate bounds eagerly using checked arithmetic.
  `TensorLayout` metadata views may use signed strides and negative slice steps
  when reachable-range validation succeeds; zero step remains invalid. Do not
  implement `PartialEq` for views.
- `ShapeVec` and `StrideVec` use `SmallVec` with inline rank capacity 8.

### Faer Integration

- Prefer zero-copy `faer::MatRef` / `faer::MatMut` views over packing data into
  temporary dense matrices. Validate shape, bounds, alignment, and aliasing
  before constructing unsafe raw faer views.
- Feed faer column-major-friendly layouts whenever possible. For faer dense
  linalg, row stride `1` is the preferred contiguous layout; generic-stride
  inputs that trigger faer performance warnings must be justified or converted
  deliberately at an explicit boundary.
- Pass faer parallelism through `CpuContext` only. Do not choose `Par::Rayon`
  or thread counts independently inside operation helpers.
- For linalg scratch space, compute scratch requirements before execution and
  allocate reusable scratch once for the operation. Avoid repeated `Vec`
  allocation in decomposition, solve, or batched inner loops.

### Performance Anti-Patterns

- Do not hand-copy near-identical dtype-specific operation bodies. Prefer
  generic helpers, sealed traits, or macros, and keep any unavoidable dtype
  dispatch isolated at the outer boundary.
- Do not allocate dense buffers when strided or backend-native access is
  available.
- Do not zero-initialize buffers that will be fully overwritten.
- Avoid per-element index multiplication in hot loops; use incremental pointer
  offsets or precomputed strides.
- Do not allocate `Vec` or other heap buffers inside hot loops. Pre-allocate
  and reuse scratch buffers.
- Do not call `Backend::plan()` or equivalent planning APIs inside execution
  loops. Pre-compute plans before the loop and pass them in.

### Performance-Sensitive Tests And Benchmarks

- Small reference tests may materialize dense tensors, but should materialize
  each full result once and compare the whole result. Avoid per-element
  re-contraction, re-evaluation, or repeated graph execution as the comparison
  mechanism.
- Long regression tests should be sized so accidental dense materialization,
  accidental `O(n^2)` graph work, or unexpected copies fail quickly while the
  intended algorithm remains cheap.
- For approximate equality, report a useful residual such as an absolute max
  error or relative norm error so performance-related failures remain
  diagnosable.
- Use release-mode benchmarks for performance claims. Prefer Criterion-style
  benchmarks for microbenchmarks, wrap benchmark inputs and outputs with
  `std::hint::black_box` where needed, and pin relevant thread counts when
  comparing CPU behavior.
- Benchmark scaling across representative tensor sizes, shapes, layouts, and
  thread counts. A single fixed-size speedup is not enough evidence for a
  performance-sensitive change.

### Cache Ownership

- Long-lived runtime/compiler caches must be owned by `Engine` or another
  explicit top-level runtime object, not hidden in thread-local/global state or
  buried in backend internals.
- Every cache must have a bounded default, a user-facing way to configure that
  bound, and a user-facing way to clear it.
- Every cache must expose user-facing introspection for the number of retained
  entries and retained bytes. Report retained bytes as the cache's owned/logical
  payload estimate, not operating-system RSS or allocator arena usage.
- Top-level runtime objects that own multiple caches must provide aggregate
  clear and aggregate stats APIs in addition to cache-specific controls.
- Backend resource pools such as buffer pools may live on the backend, but they
  still need explicit limit/clear controls, stats APIs, and documentation.
- Do not add a new cache without documenting its owner, lifetime, default
  capacity, memory behavior, entry/byte accounting, and
  clear/configuration/stats path.

### CPU Threading Contract

- For faer-backed CPU ops, `CpuContext` is the single source of truth for thread-pool policy.
- Do not derive faer parallelism independently inside individual ops or helper functions.
- Execute faer-backed work only inside `ctx.install(...)` so the owned rayon context is preserved.
- Use `Par::Seq` for one-thread contexts and `Par::rayon(0)` for multi-thread contexts so faer follows the current `CpuContext`.

### GPU Backend Contract

- Before touching CubeCL/GPU backend code, read
  [`docs/design/gpu-backend-design.md`](docs/design/gpu-backend-design.md).
- That document is the developer-facing source for CubeCL kernel ownership,
  runtime shape/stride metadata conventions, launch configuration rules, and
  device transfer behavior. Any change to those conventions must update that
  document in the same PR.
- CUDA `eig` (non-symmetric eigenvalue decomposition, LAPACK `dgeev`) is not
  provided by cuSOLVER. `CubeclBackend::eig` returns `BackendFailure`; users
  must explicitly download to CPU and compute via `CpuBackend::eig`.

## Documentation Policy

### Source of Truth

- **Source code** is the source of truth for internal design (op catalog, backend contract, AD rules, compilation pipeline).
- **Online docs** are user-facing only — how to use the explicit runtime, AD,
  tensor, GPU, and standard operation crates.
- **AGENTS.md** is the entry point for developers and AI agents. It contains pointers to source code locations.
- Do NOT duplicate source-code-level information in online docs. If it can be learned by reading the source, put a pointer instead of a copy.
- Development assumes AI agentic coding. Keep machine-readable sources (code + doc comments) authoritative.

### User-Facing Docs

- User docs target PyTorch/JAX users who interact with public direct crates
  such as `tenferro-runtime`, `tenferro-ad`, `tenferro-gpu`,
  `tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft`.
- Imports must use public direct crates such as `tenferro_runtime`,
  `tenferro_ad`, `tenferro_gpu`, and standard extension crates. Do not
  reference internal crates (`tenferro-internal-ops`, `computegraph`, etc.) in
  user-facing docs.
- Do NOT expose internal jargon (Fragment, StableHLO, ExecOp, ValRef, etc.) in user-facing pages.
- Provide PyTorch/JAX equivalents when introducing tenferro concepts.

### User-Facing Code Snippet Consistency

- Non-trivial user-facing code snippets must have an executable source of truth:
  prefer including a checked example, test, or doctest instead of copying code
  into Markdown by hand.
- If a Markdown page must contain a copied snippet, add an automated sync or
  extraction check that fails when the snippet drifts from the executable
  source.
- Guide code that demonstrates a workflow should compile in CI. When runtime
  execution requires special hardware or external libraries, CI must still
  compile-check the example with the required feature flags, and the guide must
  document the command that runs the example on a correctly configured machine.
- CPU and GPU workflow examples should include meaningful assertions on shapes,
  dtypes, or values whenever the result is deterministic. Avoid examples that
  only print output unless the output itself is the documented behavior.
- GPU quickstart examples must use the same executable source as the guide,
  explicitly show upload/download boundaries, and assert downloaded CPU results
  for at least one supported operation.

### Doc Examples

- Doc examples (`/// # Examples`) must NOT use `ignore` or `no_run` attributes.
- Every example must compile AND run as a doctest.
- Use `compile_fail` only for examples that intentionally demonstrate compile errors.
- If an example cannot run as a doctest, refactor it until it can.

## Generic Over Scalar Type

- Use generic constructors with sealed traits instead of per-type functions.
- Bad: `TracedTensor::from_f64(...)`, `TracedTensor::from_f32(...)`, etc.
- Good: `TracedTensor::new<T: TensorScalar>(shape, data)` — type inference selects the variant.
- For dtype-polymorphic tensor operations, prefer one typed generic
  implementation plus outer dtype dispatch over per-dtype copy-pasted
  implementations.
- If Rust generics cannot express the shared structure cleanly, use a local
  macro to generate the repetitive dispatch or variant plumbing.
- Sealed traits (`TensorScalar`, `PoolScalar`, etc.) restrict APIs to the
  currently supported dtype set.

## Public API Convention

- **Unary single-output ops**: methods on `TracedTensor` (e.g., `x.exp()`, `x.reshape(shape)`)
- **Binary single-output ops**: operator overloads where natural (`&a + &b`, `&a * &b`), methods otherwise (`a.dot_general(&b, config)`)
- **Multi-output ops**: free functions (e.g., `svd(&a)`, `qr(&a)`, `eigh(&a)`)
- **Linalg ops**: free functions (e.g., `solve(&a, &b)`, `cholesky(&a)`)
- **Einsum**: free function `einsum(engine, inputs, subscripts)`
- No `traced_` prefix on methods. `TracedTensor` methods are inherently traced.
