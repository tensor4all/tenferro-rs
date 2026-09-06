# Performance Tips

Tenferro performance, layout, cache, threading, and backend contracts, kept
separate from `REPOSITORY_RULES.md` so they can be read on demand. These rules
apply on top of the shared `rules/rust/performance.md` and
`rules/rust/numerical.md` from `tensor4all-agent-rules`.

Read this file in full:

- before implementing or reviewing performance-sensitive code: tensor kernels,
  graph/compiler planning, caches, GPU kernels, benchmarks, and user-facing
  examples;
- before creating a PR that touches such code;
- when running the `audit-performance` skill or a static performance audit.

Each `## ` section is a routing unit for `scripts/repository-rules-review.py`,
which parses this file together with `REPOSITORY_RULES.md`. Section titles must
be unique across both files; new sections need an entry in that script's
routed or human-only lists. The `Audit hints` under each section are for static
audits: `Detect` lists source patterns worth inspecting, `Fix` the expected
direction. A hint match is evidence to inspect, not a finding by itself.

## Audit Procedure

Human/process protocol. This section is intentionally not routed to the
diff-scoped review bot.

1. Scope: audit the requested paths, or with `full` the crates under `crates/`
   and `ext/`, then `benches/`, examples, and doc snippets. Skip `docs/plans/`
   and `docs/worklogs/`.
2. For each section below, search the scope for its `Detect` patterns and read
   the surrounding code. Respect an `// INVARIANT:` marker that explains the
   pattern unless the explanation is wrong.
3. Report each finding as `file:line`, the section title, the evidence, and the
   `Fix` direction. Group findings that share a root cause.
4. Findings are static rule violations. Never claim a speedup or slowdown
   without a measurement, and start any optimization through the
   Performance-Gated Experiment Protocol.
5. Cross-check the Public Boundary Safety Audits and Unsafe Code Boundary
   sections of `REPOSITORY_RULES.md` when a finding touches unsafe code.

## Performance-Sensitive Safety Contracts

- Potentially dangerous operations kept for performance (raw scratch-buffer
  acquisition, unchecked indexing after validation, raw pointer arithmetic,
  backend-native view construction) carry a nearby one-line `// INVARIANT:`
  comment (see Invariant Markers) explaining why they are valid, even when
  technically correct, so later agents and reviewers do not "fix" a false
  positive with hidden copies, repeated checks, or unconditional
  initialization in a hot path.
- Buffer-pool and scratch-buffer APIs distinguish full-overwrite callers from
  read-before-write callers. Do not fix stale or uninitialized reads with
  unconditional zero-fill in shared hot-path acquisition; expose an explicit
  zeroed/initialized path, keep raw acquisition unsafe and documented for
  full-overwrite kernels only, and add regression coverage for both.
- While a backend still exposes a cloneable owner-like `Arc` buffer, an
  in-place write requires a local uniqueness proof such as `Arc::get_mut` or
  exclusive runtime ownership for the whole operation. This applies to that
  legacy owner representation, not to lifetime-only root retention. Under the
  final storage contract, an `Arc<RootResource>` may be shared by disjoint
  claims, read-only retained records, or retirement records; its strong count
  is neither write authority nor a reason to reject a valid write. Write
  authority comes only from provenance-carrying `StorageMut` derived from an
  exclusive Rust borrow (or a fresh output), with checked span and injectivity
  proofs. Add mutation-after-handoff and disjoint-claim tests where an
  optimization preserves aliases or views that a previous implementation
  copied.

Audit hints:

- Detect: raw scratch acquisition, unchecked indexing, raw pointer arithmetic,
  or backend-native view construction without a nearby `// INVARIANT:`;
  unconditional zero-fill added to a shared hot-path acquisition; in-place
  writes through a cloneable `Arc` buffer without a uniqueness proof.
- Fix: add the marker with the proof, or route through the explicit
  zeroed/initialized path; restore write authority to `StorageMut` derived
  from an exclusive borrow, and add the aliasing regression tests.

## Complexity Budget

- No accidental `O(n^2)` behavior in graph construction, metadata propagation,
  key hashing/equality, compilation, or execution scheduling. An intentional
  quadratic algorithm documents the bound or tradeoff with an `// INVARIANT:`
  marker (see Invariant Markers).
- Do not repeatedly clone, hash, format, or scan whole graph histories,
  metadata scope lists, tensor input maps, or structural keys inside
  per-node/per-op loops. Prefer stable IDs, interning, cached fingerprints
  with exact equality checks, or persistent/shared data structures.
- When optimizing compiler or graph-build overhead, measure scaling across
  increasing graph sizes, not one fixed case.

Audit hints:

- Detect: clone, hash, format, `contains` on a `Vec`, or linear scan of graph
  history, scope lists, input maps, or structural keys inside per-node or
  per-op loops; nested loops over node counts without an `// INVARIANT:`.
- Fix: stable IDs, interning, cached fingerprints, `HashMap`/`HashSet`
  lookups, or persistent structures; measure scaling across graph sizes.

## Materialization And Copies

- Production tensor paths must not silently materialize dense temporaries
  whose memory or time scales with an unconstrained product of tensor
  dimensions. Dense/reference implementations are allowed only when the API is
  explicitly named and documented as dense, reference, or debug behavior.
- Avoid dense copy-in/copy-out around operations that can consume strided
  views, borrowed slices, backend-native buffers, or metadata-only layout
  changes.
- No hidden CPU-GPU transfers. Callers explicitly upload/download tensors;
  execution pipeline internals may move constants or scalar metadata only
  where the backend contract documents it.
- When an output contiguity contract, backend limitation, or external ABI
  boundary requires a copy or materialization, make that boundary explicit in
  the implementation and cover it with tests.

Audit hints:

- Detect: `to_vec`, `collect::<Vec<_>>`, `to_dense`, `clone` of tensor data,
  or a fresh output allocation on a production path whose size is a product
  of tensor dimensions; `to_host`/`to_device` inside library operations.
- Fix: strided views, borrowed slices, metadata-only layout changes, or
  backend-native buffers; make any required copy explicit and tested.

## Dense Layout And Linear Algebra

- tenferro uses column-major (Fortran order) dense storage: the leftmost
  dimension has the smallest stride and varies fastest.
- Owned runtime tensors are compact column-major only. Arbitrary strides,
  offsets, transposes, slices, and reverse views live on
  `TypedTensorView`/`TypedTensorViewMut` or metadata-only layout values until
  an explicit same-placement canonicalization boundary.
- Public flat-buffer constructors, exports, examples, FFI contracts, and docs
  state or preserve column-major semantics.
- No row-major compatibility shims or hidden row-major round-trips in library
  code. Convert naturally row-shaped external data privately at that explicit
  boundary.
- For batched GEMM-style layouts, put contraction/compute dimensions on the
  left and batch dimensions on the right, keeping each batch slice contiguous
  in column-major storage.
- Use existing tensor/backend abstractions for GEMM, einsum, and dense linear
  algebra. Do not reimplement linalg kernels in downstream layers when the
  CPU/GPU backend owns the operation.

Audit hints:

- Detect: row-major index arithmetic (`i * cols + j`), row-major round-trips,
  batch dimensions placed left of compute dimensions, or hand-written GEMM,
  solve, or decomposition loops outside the owning backend.
- Fix: column-major strides from the layout value, batch on the right, and
  the existing backend operation.

## Range Checks And Slicing

- Public indexing and slicing APIs validate rank, bounds, steps, output shape,
  and empty/singleton boundary behavior at the API or planning boundary.
- Layout metadata and runtime typed views may use signed strides and negative
  slice steps when reachable-range validation proves every logical element
  maps inside the backing allocation. Zero step remains invalid. Do not reject
  negative strides solely for being negative. Narrower adapter APIs may
  document stricter limits, explicitly at the API boundary.
- After validation, hot loops and kernels should not repeat range checks per
  element; carry validated shape/stride/offset metadata inward.
- Prefer safe Rust patterns that let LLVM eliminate bounds checks: iterate over
  slices directly, slice once before the loop, use `chunks_exact` when the
  chunk size divides the length, and add pre-loop assertions for validated
  index ranges. Unchecked indexing is a last resort.
- Prefer metadata-only slices, strided views, or backend-native slice kernels
  over dense copies. If a slice must allocate, document and test the reason.
- Slice, reshape, transpose, gather, scatter, pad, concatenate, and reverse
  preserve column-major shape/stride/offset semantics.
- Unchecked indexing or unsafe pointer access after validation keeps the
  invariant close to the unsafe block, with tests for full range, empty or
  singleton slices, lower and upper boundaries, out-of-range errors, rank
  mismatch, and non-contiguous slices.

Audit hints:

- Detect: per-element `assert!`, `checked_*`, or indexed `[]` access on
  validated data inside hot loops; unchecked access without a nearby
  invariant; slices that allocate without a documented reason; stride
  rejection based only on sign.
- Fix: validate once at the boundary, carry metadata inward, iterate slices
  directly, and keep the boundary tests listed above.

## Faer Integration

- Prefer zero-copy `faer::MatRef` / `faer::MatMut` views over packing into
  temporary dense matrices. Validate shape, bounds, alignment, and aliasing
  before constructing unsafe raw faer views.
- Feed faer column-major-friendly layouts. For dense linalg, row stride `1` is
  the preferred contiguous layout; generic-stride inputs that trigger faer
  performance warnings must be justified or converted deliberately at an
  explicit boundary.
- Pass faer parallelism through `CpuContext` only; never choose `Par::Rayon`
  or thread counts inside operation helpers.
- Compute linalg scratch requirements before execution and allocate reusable
  scratch once per operation. No repeated `Vec` allocation in decomposition,
  solve, or batched inner loops.

Audit hints:

- Detect: packing into temporary dense matrices before a faer call;
  `Par::rayon(...)`/`Par::Seq` chosen inside an op helper; `Vec` scratch
  allocated per decomposition, solve, or batch iteration.
- Fix: zero-copy `MatRef`/`MatMut` after validation, parallelism from
  `CpuContext`, scratch sized once per operation.

## Performance Anti-Patterns

- Do not hand-copy near-identical dtype-specific operation bodies. Use generic
  helpers, sealed traits, or macros, and isolate unavoidable dtype dispatch at
  the outer boundary.
- Do not allocate dense buffers when strided or backend-native access is
  available.
- Do not zero-initialize buffers that will be fully overwritten.
- Avoid per-element index multiplication in hot loops; use incremental pointer
  offsets or precomputed strides.
- Do not allocate `Vec` or other heap buffers inside hot loops; pre-allocate
  and reuse scratch.
- Do not call `Backend::plan()` or equivalent inside execution loops;
  pre-compute plans and pass them in.

Audit hints:

- Detect: near-identical dtype-specific bodies; `vec![0 ...]` or `Vec::new()`
  inside loops; zero-fill before a full overwrite; index multiplication per
  element; `plan(` or equivalent called per execution.
- Fix: generic helpers or macros with dispatch at the outer boundary, hoisted
  scratch, incremental offsets, and precomputed plans.

## Performance-Sensitive Tests And Benchmarks

- Small reference tests may materialize dense tensors, but should materialize
  each full result once and compare the whole result. No per-element
  re-contraction, re-evaluation, or repeated graph execution as the comparison
  mechanism.
- Long regression tests should be sized so accidental dense materialization,
  `O(n^2)` graph work, or unexpected copies fail quickly while the intended
  algorithm stays cheap.
- For approximate equality, report a useful residual such as absolute max
  error or relative norm error.
- Use release-mode benchmarks for performance claims. Prefer Criterion-style
  microbenchmarks, wrap inputs and outputs with `std::hint::black_box` where
  needed, and pin thread counts when comparing CPU behavior.
- Benchmark scaling across representative sizes, shapes, layouts, and thread
  counts. A single fixed-size speedup is not enough evidence.

Audit hints:

- Detect: element-wise reference loops that re-run contraction or graph
  execution per element; debug-mode timing claims; missing `black_box`;
  benchmarks at one size or thread count only.
- Fix: materialize once and compare whole results, release-mode Criterion
  benchmarks with pinned threads across representative sizes.

## Performance-Gated Experiment Protocol

Human/process protocol. This section is intentionally not routed to the
diff-scoped review bot.

- Performance candidates found by static/source audit alone pass a
  need-before-implementation gate before code changes start: measure the
  candidate path's share of an end-to-end workload or another
  issue-predeclared representative workload. If the share is not meaningful
  under the predeclared threshold, record the measurement or argument in the
  issue and close or defer without implementation. A microbenchmark of the
  helper is useful after the need is established, but alone proves only
  effect, not that the optimization is worth its semantic, aliasing, cache, or
  maintenance risk.
- Before running a candidate, record the baseline commit, candidate commit,
  benchmark source, build profile, hardware and affinity configuration,
  provider/thread settings, complete case list, comparison statistic,
  acceptance threshold, repetition policy, and host-noise observables and
  thresholds. Candidate results must not influence these choices.
- Run the complete baseline/candidate suite as one paired experiment. Do not
  selectively retry, omit, replace, or promote individual favorable cases.
- If a predeclared host-noise or validity gate fails, classify the entire
  paired experiment as `INCONCLUSIVE`. Reconsideration requires a complete
  paired rerun under the same protocol.
- Record every measured case, confidence interval, validity observation, and
  regression in the worklog. A negative or inconclusive primary result is
  evidence and must not be rewritten as success because secondary cases
  improved.
- Promote a performance-gated change only when its predeclared primary gate
  and all required non-regression/correctness gates pass. Do not relax
  thresholds, redefine the primary metric, or add post-hoc exclusions after
  seeing the candidate.

Audit hints:

- Detect: an optimization PR or issue without a recorded end-to-end share,
  predeclared thresholds, or a complete paired run; selective retries or
  post-hoc exclusions in the worklog.
- Fix: record the need measurement and protocol in the issue before code
  changes; report negative or inconclusive results as such.

## Cache Ownership

- Long-lived runtime/compiler caches are owned by `Engine` or another explicit
  top-level runtime object, not hidden in thread-local/global state or buried
  in backend internals.
- Every cache has a bounded default, a user-facing way to configure that
  bound, and a user-facing way to clear it.
- Every cache exposes user-facing introspection for retained entries and
  retained bytes, reported as the cache's owned/logical payload estimate, not
  RSS or allocator arena usage.
- Top-level runtime objects owning multiple caches provide aggregate clear and
  aggregate stats APIs in addition to cache-specific controls.
- Backend resource pools such as buffer pools may live on the backend but
  still need explicit limit/clear controls, stats APIs, and documentation.
- Backends owning resource pools or runtime contexts, including `CpuBackend`'s
  buffer pool and `Arc<CpuContext>` and CUDA backends' runtime client/context,
  are construct-once-and-reuse values. Examples should bind the backend once and
  reuse it across related operations; they must not present per-call
  `CpuBackend::new()` or GPU backend construction chained into an operation as
  the normal idiom. A genuinely standalone single-op example need not invent an
  unrelated long-lived backend variable.
- Do not add a cache without documenting its owner, lifetime, default
  capacity, memory behavior, entry/byte accounting, and
  clear/configuration/stats path.

Audit hints:

- Detect: `OnceLock`, `lazy_static`, `thread_local!`, or a `static` map used
  as a cache; a cache type without bound, clear, and stats APIs; examples
  constructing `CpuBackend::new()` or a GPU backend per call.
- Fix: own the cache from `Engine` or the top-level runtime object with
  bound/clear/stats controls and documentation; bind the backend once.

## CPU Threading Contract

- For faer-backed CPU ops, `CpuContext` is the single source of truth for thread-pool policy.
- Do not derive faer parallelism independently inside individual ops or helpers.
- Execute faer-backed work only inside `ctx.install(...)` so the owned rayon context is preserved.
- Use `Par::Seq` for one-thread contexts and explicit `Par::rayon(n)` from the
  configured `CpuContext` degree for multi-thread contexts. Do not derive the
  policy from an ambient Rayon pool during plan or session setup.
- Tensor-sized strided CPU kernels that are not provider-owned also run inside
  `CpuContext::install(...)`, so Rayon-backed `strided-kernel` work uses the
  backend's owned pool. BLAS/LAPACK provider-owned threading remains
  controlled by provider variables such as `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`.

Audit hints:

- Detect: faer or `strided-kernel` work outside `ctx.install(...)`;
  `rayon::current_num_threads`, `ThreadPoolBuilder`, or `Par::rayon(0)` used
  to derive policy; thread counts chosen inside op helpers.
- Fix: take the degree from `CpuContext` and run inside its installed pool;
  leave provider-owned threading to the provider variables.
