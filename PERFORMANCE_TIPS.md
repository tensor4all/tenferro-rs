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
- Audit owned, compact borrowed, offset/strided borrowed, and preallocated-output
  paths together on CPU and GPU. `TensorRead::View` is not a contiguity test:
  do not pack an already supported layout merely to call an owned-input API.
  Route owned inputs through borrowed execution where possible; allocating and
  `_into` operations should share the output-writing kernel, not allocate a
  temporary result and copy it into the caller's destination.
- A `*_read` hook is the route that avoids the staging copy, so a backend that
  inherits a materializing default must override it as soon as its kernel can
  consume the read's strides, and the default must say at its own definition
  that it stages. Do not work around a materializing hook at the call site with
  `to_contiguous_read` or an owned-input API: fix the hook (#2214). A read hook
  with no stride-consuming kernel keeps the materializing behaviour as an
  explicit, documented boundary and must not back a user-facing form that
  silently stages.
- Regression evidence for a read hook must separate the operation's required
  output storage from operand staging. Allocation counts alone do not separate
  them: a warm buffer pool can serve the staging buffer without touching the
  allocator. Count the canonicalization itself (a counter or spy on
  `to_contiguous_read`), or the retained pool bytes, and keep a negative control
  that fails when the hook falls back to the materializing default.
- Trace copies across eager/runtime adapters, extension dispatch, and provider
  preparation. Reuse an existing read hook instead of packing before it, and
  avoid packing followed by another full copy into vendor work storage. Keep
  required independent ownership, destructive-input protection, alias checks,
  and asynchronous storage retention; do not replace these with unsafe aliases.
- Regression checks for removed materializations must cover data movement as
  well as numerical results: use existing allocation/copy counters, a dispatch
  spy, or a focused device trace. Include compact views and a noncompact/offset
  case, input preservation, and `_into` where supported. CPU-only numerical
  tests do not establish CUDA copy behavior.

Audit hints:

- Detect: `to_vec`, `collect::<Vec<_>>`, `to_dense`, `clone` of tensor data,
  or a fresh output allocation on a production path whose size is a product
  of tensor dimensions; `to_host`/`to_device` inside library operations;
  `as_tensor()`/`TensorRead::View` branches that unconditionally materialize;
  read-into helpers that call allocating operations; repeated canonicalization
  and work-buffer copies across adjacent layers.
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

- Batched CPU linalg (LU factor, prepared LU solve, triangular solve, full
  pivot LU, eigh, SVD, QR, Cholesky, solve) runs as one tight loop per call:
  compute scratch requirements before execution, query the provider workspace
  once per call, allocate scratch once and reuse
  it across the batch, and write each matrix result directly into the batch
  output. Do not build per-matrix `TypedTensor`s, call a `*_2d` helper that
  allocates per matrix, take pool buffers that are never returned, or allocate
  `Vec`s for pivots or permutations inside the batch loop. The eager solve and
  QR rewrite (26abf5ec) is the reference pattern. Small matrices at large
  batch (for example 2x2 with batch 1024) are the regression case, because
  per-matrix overhead dominates there while LAPACK time is negligible
  ([#1878](https://github.com/tensor4all/tenferro-rs/issues/1878)).

Audit hints:

- Detect: row-major index arithmetic (`i * cols + j`), row-major round-trips,
  batch dimensions placed left of compute dimensions, or hand-written GEMM,
  solve, or decomposition loops outside the owning backend.
- Detect: workspace queries (`lwork = -1` style calls), `from_host_vec`,
  `pooled_copy`, `pooled_zeroed`, per-matrix tensor construction, or `Vec`
  allocation inside a batched linalg loop, including through
  `batched_multi`/`batched_binary_result`-style helpers.
- Fix: column-major strides from the layout value, batch on the right, and
  the existing backend operation. Hoist the workspace query and scratch out of
  the batch loop and write into the batch output in place.

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
- Pass faer parallelism through the operation's placement token only
  (`Exec::external` hands the budget over as a plain width; `Par::Seq` for a
  width of one, else `Par::rayon(width)`); never choose `Par::rayon` or thread
  counts inside operation helpers.
- Scratch and batching follow the batched linalg rule in Dense Layout And
  Linear Algebra, which applies to faer and LAPACK providers alike.

Audit hints:

- Detect: packing into temporary dense matrices before a faer call;
  `Par::rayon(...)`/`Par::Seq` chosen inside an op helper; `Vec` scratch
  allocated per decomposition, solve, or batch iteration.
- Fix: zero-copy `MatRef`/`MatMut` after validation, parallelism from the
  operation's placement token, scratch sized once per operation.

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

- Resolve a runtime operation enum, dtype, or flag once per call or worker
  range, never per element. A closure that matches on a runtime op inside a
  tensor-sized loop defeats vectorization; this defect class hid in strided
  erased entries behind tenferro's delegation
  ([strided-rs#269](https://github.com/tensor4all/strided-rs/issues/269)).
- Trace lowering and tracked eager AD must not route a primal op around the
  fast path that untracked eager uses. Splitting an op into prepared
  primitives (for example `solve` into `LuFactor` plus `LuSolvePrepared`) to
  expose residuals needs either a fused primal kernel that also emits the
  residuals or a lowering that keeps the fused op when the intermediate has no
  other consumer. Trace and tracked eager primal time for such an op stays
  within 1.25x of untracked eager at the benchmark sizes.

Audit hints:

- Detect: near-identical dtype-specific bodies; `vec![0 ...]` or `Vec::new()`
  inside loops; zero-fill before a full overwrite; index multiplication per
  element; `plan(` or equivalent called per execution.
- Detect: `match op` or dtype dispatch inside a per-element closure passed to
  a traversal; a trace or AD lowering that emits several primitives for an op
  that eager executes as one fused provider call.
- Fix: generic helpers or macros with dispatch at the outer boundary, hoisted
  scratch, incremental offsets, and precomputed plans. Keep the fused primal
  op in lowering and attach residuals to it.

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
- Audit AD residual reuse end to end, not just the rule's residual declaration:
  verify that eager forward's saved values actually reach backward execution.
  Distinguish eager forward-plus-backward, backward-only, and prepared trace
  timing. Use provider-call counters or execution spies to detect repeated
  decompositions and per-target replay of shared derivative work; numerical
  gradient tests alone do not establish reuse.
- Do not default to recomputing `exp`, `tanh`, or other elementwise residuals
  merely because they are cheaper than decompositions. Preserve only the
  declared required residuals. Recomputation must follow an explicit checkpoint
  policy or a measured memory/fusion tradeoff, with its scope documented and
  first- and higher-order derivatives preserved. Cover both dropped forward
  output handles and multiple differentiated inputs when testing reuse.

- Thread settings in benchmark harnesses are enforced and verified, not only
  requested: build the `CpuEngine` pool with the requested degree, assert
  `engine.num_threads()` at startup, and fail on conflicting thread environment
  variables. A `--num-threads 1` flag that left an ambient pool running
  multi-threaded once produced a 7x wrong one-thread row.
- For tensor-sized CPU cases, compare one and four threads in the same run: a
  four-thread time that is not faster than the one-thread time is a finding,
  even when the one-thread row matches the reference.
- Every public op family has benchmark rows. Adding a public op, or routing an
  existing op through a new kernel (for example max/min reductions moving to
  strided), adds its rows to tenferro-benchmark in the same change or links
  the benchmark PR. When the op delegates to strided, the matching rows also
  exist in strided-rs-benchmark-suite so the defect is caught at the strided
  level first. Cover all reduction ops (sum, prod, max, min) over all axes and
  single axes, and batched linalg at small matrix sizes.
- Benchmark each execution mode that users can select (untracked eager,
  tracked eager, trace) for the same op, so a lowering that bypasses the eager
  fast path shows up as a mode gap.
- Build parallelism and measured thread count are different settings. Compile
  with an explicit job count (`cargo bench -j 16`, or `CARGO_BUILD_JOBS=16`);
  leaving the build at one job spends minutes in serial compilation without
  making the measurement any more single-threaded than the thread environment
  already does. The measured binary itself must run with a pinned thread count.
- Pin the measured process to a core and check that core is idle immediately
  before and after each measurement (a `/proc/stat` busy fraction over a few
  seconds is enough). Unpinned background jobs on a shared host migrate between
  cores and can inflate one run of an entire target; prefer short per-case runs
  (one filtered case at `--sample-size 100` is seconds) over long target-wide
  runs. A record of how the run was made is part of the evidence; note in the
  work log whether the build parallelism matched the recorded manifest.
- Read the shape of a slowdown before claiming a cause. A constant absolute
  delta across sizes is a per-operation, per-entry or per-invocation cost; a
  uniform multiplicative factor across cases the change cannot affect is host
  contention. A same-binary A/A run (the same source measured twice, minutes
  apart) gives the noise floor that separates the two, and Criterion's own
  `change:` line against the previous run in the same target directory is an
  independent check.
- Keep before/after harness identity explicit: the arm list, the case names and
  the Criterion settings are part of a baseline's identity. When the arm list
  changes, recapture the baseline with the new harness rather than comparing
  across it, and record which commit produced each side.

Audit hints:

- Detect: element-wise reference loops that re-run contraction or graph
  execution per element; debug-mode timing claims; missing `black_box`;
  benchmarks at one size or thread count only.
- Detect: AD rules declare residuals but eager execution rebinds only original
  inputs and replays their producer graph; separate backward executions per
  leaf; unconditional elementwise rematerialization without measured benefit.
- Detect: a thread flag parsed but not used to build the backend context;
  missing startup verification of the effective thread count; a public op or
  reduction variant with no tenferro-benchmark row; an op benchmarked in one
  execution mode only.
- Fix: materialize once and compare whole results, release-mode Criterion
  benchmarks with pinned threads across representative sizes. Connect saved
  residual values to backward execution and add call-count regression tests;
  keep intentional checkpoint/rematerialization cases separately labeled.

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
- Summarize the decision, primary result, validity, and remaining limitations
  in the work log. Retain every measured case, confidence interval, validity
  observation, regression, and reproduction-critical setting in the experiment
  results, linked from the work log; do not copy the full results or command
  history into it. A negative or inconclusive primary result is evidence and
  must not be rewritten as success because secondary cases improved.
- Promote a performance-gated change only when its predeclared primary gate
  and all required non-regression/correctness gates pass. Do not relax
  thresholds, redefine the primary metric, or add post-hoc exclusions after
  seeing the candidate.
- Record the host observables that decide validity next to the predeclared
  thresholds: the pinned core, its idle observation before and after each
  measurement, and the load at start. A paired experiment whose pinned core was
  busy during either side is `INCONCLUSIVE`, not a candidate result.
- When a claim is about a countable mechanism (pool entries, provider calls,
  kernel launches), count it rather than infer it from timing. A pool entry can
  be counted on the calling thread around the operation; a probe or lookup that
  cannot succeed should be shown to cost no entry rather than assumed free.

Audit hints:

- Detect: an optimization PR or issue without a recorded end-to-end share,
  predeclared thresholds, or a complete paired run; selective retries or
  post-hoc exclusions in the linked experiment evidence.
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
- Engine resource pools such as buffer pools may live on the engine but still
  need explicit limit/clear controls, stats APIs, and documentation.
- Engines owning resource pools or runtime contexts, including `CpuEngine`'s
  buffer pool, workspace and caller-supplied `Pool`, and CUDA engines'
  runtime client/context, are construct-once-and-reuse values. Examples should
  bind the engine once and reuse it across related operations; they must not
  present per-call `CpuEngine::new()`/`CpuEngine::serial()` (or GPU engine
  construction) chained into an operation as the normal idiom. A genuinely
  standalone single-op example need not invent an unrelated long-lived engine
  variable.
- Do not add a cache without documenting its owner, lifetime, default
  capacity, memory behavior, entry/byte accounting, and
  clear/configuration/stats path.

Audit hints:

- Detect: `OnceLock`, `lazy_static`, `thread_local!`, or a `static` map used
  as a cache; a cache type without bound, clear, and stats APIs; examples
  constructing `CpuEngine::new()`/`CpuEngine::serial()` or a GPU engine per call.
- Fix: own the cache from `Engine` or the top-level runtime object with
  bound/clear/stats controls and documentation; bind the engine once.

## CPU Threading Contract

- A CPU engine owns exactly one `threadteam::Pool`, or none for
  `CpuEngine::serial()`. The caller builds the pool (with `Pool::builder()`, which
  `tenferro-cpu` re-exports) and hands it over; tenferro never starts a hidden or
  global pool, keeps no process-wide registry of engines, and does not
  arbitrate between callers. `&mut engine` is the exclusivity mechanism.
  Concurrent computations use separate engines and separate pools, each on its
  own calling thread.
- Only `threadteam` starts tasks. Where CPU work runs and how many tasks run at
  once is decided by one borrowed token, `threadteam::Exec`: a pool reference
  plus a budget. It is neither `Clone` nor `Copy`, every parallel entry
  consumes it, and a split (`fork`, `for_ranges`) gives each child a share of
  the budget, so the tasks running at once under one operation never exceed its
  budget at any nesting depth. tenferro has no raw Rayon call, no thread-local
  placement state and no second clamp: `Exec::new` clamps the budget once, to
  the pool width.
- The thread width is decided before the pool is built and never changed at
  run time. Do not derive it from an ambient pool, do not build a second pool
  inside an operation, and do not install or restore an execution context per
  operation. `CpuEngine::num_threads()` reports the fixed budget.
- Every operation starts on the calling thread and mints one token from the
  engine's pool. Kernels take it by value as their **last** parameter; a kernel
  that runs several parallel phases takes `reborrow()` for each of them, and
  one that runs independent parts forks it. Plans never store a token; replay
  takes it per call. A serial engine mints a serial token: no pool is started
  or entered. A small operation never enters the pool because a serial token, a
  budget of one and a single task run inline.
- cpueinsum/tcontract and tlinalg receive the engine's provider bundle
  (`CpuEngine::exec`, a `tcontract_exec::Exec`: token, workspace, batch
  policy) by `&mut` and fork the token for their lanes. They never call back
  into tenferro from a lane. Direct downstream faer calls take their width from
  the same token with `Exec::external`.
- For faer-backed CPU ops the token is the single source of truth for
  thread-pool policy. Choose the faer width from the shared work model
  **before** any pool entry and hand it over with `Exec::limit(..).external(..)`;
  never choose `Par::rayon` or a thread count inside an operation helper. Do
  not derive faer parallelism independently. How faer divides the width it
  receives is faer's concern.
- The token is correct when the operation is called from a plain thread, from a
  worker of the engine's own pool (the entry runs in place) and from a worker of
  a different pool. A kernel that needs every worker at once (a barrier team)
  asks `Exec::can_broadcast` and takes a barrier-free path when it is refused.
- Pinning is the caller's job. Pin the calling thread or the pool's workers
  (`PoolBuilder::on_start`) when measuring or placing work; tenferro does not
  narrow and restore CPU masks per operation. Do not assign one CPU per worker:
  vendor teams inherit the creating worker's mask, so a one-CPU worker mask
  silently confines the whole provider team to one CPU (measured: 3-20x slower
  dgemm/LU paths at four threads). Provider thread counts remain controlled by
  the provider variables below, not by worker affinity.
- `PoolBuilder` reserves 16 MiB per worker by default rather than the
  `std::thread` default of 2 MiB, because provider kernels recurse with large
  private frames: a `NUM_THREADS=64` OpenBLAS build keeps about 541 KiB per
  `dgetrf_parallel` level, which overflowed a 2 MiB worker at n>=256. Lower it
  with `stack_size` only for pools that never run provider code.
- tenferro does not own an outer fan-out for ordinary numerical work: lane
  selection for batched and grouped work belongs to cpueinsum and tlinalg, which
  fork the token they receive. tlinalg-blas takes no token: the vendor owns its
  threading. No placement promise is made for threads a vendor creates.
- Vendor BLAS/LAPACK is called from the coordinator thread; tenferro bounds
  only the token it passes to its own lower libraries.
- Tensor-sized strided CPU kernels that are not provider-owned also run on the
  engine's pool through the token. BLAS/LAPACK provider-owned threading remains
  controlled by provider variables such as `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`.
- The invariants are checked mechanically: `deny.toml` (`cargo deny check bans`)
  rejects a direct `rayon` dependency anywhere in the workspace;
  `scripts/check-no-public-rayon.sh` (`cargo public-api`) fails when a `rayon::`
  path appears in `tenferro-cpu`'s public API; tests read the
  `threadteam::ledger` through `CpuEngine::take_ledger_*` (feature `ledger`)
  for `max_live <= budget` and for zero pool installs on small operations, and
  a test binary of its own asserts `ledger::global_pool_unstarted()`.

Audit hints:

- Detect: a hidden or global pool, a direct `rayon` dependency or `par_iter`, a
  pool or engine built inside an operation, a second `Pool::builder()` per call,
  thread width derived from an ambient pool, a per-operation context install, a
  token stored in a plan or struct, a token passed twice instead of
  `reborrow`/`fork`, a lock held across provider work, `Par::rayon(0)`, or
  per-worker single-CPU affinity that provider-created threads would inherit.
- Fix: build one `Pool`, hand it to one engine, take `&mut` for the operation,
  pass the minted token down by value, split it with `fork`/`for_ranges`, and
  leave provider-owned threading to the provider variables.
