# Issue #1716: caller-managed CPU admission

## Summary

Implemented explicit caller-managed admission for caller-owned CPU executors.
Caller-managed domains declare no CPU set, bypass process-wide CPU-set
arbitration, retain a per-domain public-entry guard, and execute faer/native work
only through the supplied executor. Cooperative external domains retain their
existing placement and arbitration behavior.

## Context read

- Issue #1716 and its acceptance criteria
- `AGENTS.md`, `REPOSITORY_RULES.md`, workspace `CODING_RULES.md`, and the
  relevant shared Rust, performance, documentation, and testing rules
- `docs/design/cpu-backend-execution.md`
- `docs/design/execution-engine-provider-architecture.md`
- `docs/guides/cpu-execution.md`
- `docs/guides/parallelism-and-caching.md`
- CPU domain executor, resource domain, backend, arbiter, provider, provider
  capability, engine, and existing external-managed tests
- Rayon `ThreadPool::install` / registry same-pool entry behavior from the
  installed Rayon source

## Decisions

- Kept `ExternalCpuDomain::new` as the cooperative CPU-set constructor and added
  `new_caller_managed` rather than introducing a parallel domain type.
- Added `CpuAdmissionMode` and made placement, CPU-set, and placement-guarantee
  diagnostics optional. Caller-managed domains do not fabricate affinity facts.
- Added `CpuBackend::for_domain(CpuDomainId)` because placement-free domains
  cannot be selected through `CpuPlacement`.
- Added a thin `RayonCpuDomainExecutor` adapter that retains exactly one
  caller-owned `Arc<rayon::ThreadPool>`; the generic executor trait remains the
  primary seam.
- Used a per-domain RAII active-entry flag to reject recursive or simultaneous
  public entry into one caller-managed domain across all caller-pool workers.
  Distinct domains have independent flags and never enter `ResourceArbiter`.
- Selected faer explicitly for caller-managed registries. Provider validation
  has a caller-managed branch with no advisory/all-allowed bypass and rejects
  external workers, missing placement control, or uncontrolled thread counts
  before returning a backend.

## Alternatives rejected

- Fabricated `AllAllowed` placements: they would preserve the overlap conflict
  and misreport a placement contract the caller did not make.
- A global registry of external pools: unnecessary process-global state and
  contrary to the accepted issue.
- Worker TLS installation on caller-created Rayon pools: start handlers cannot
  be retrofitted, and a per-domain entry guard enforces the public re-entry
  contract without reconstructing the pool.
- Allowing BLAS/LAPACK and documenting weaker isolation: provider-owned workers
  cannot satisfy the supplied-executor boundary.

## Review gates

- Pre-implementation design: `reviewer-flash-opencode-go` requested changes to
  provider validation, cross-worker re-entry, domain-ID selection, and optional
  placement scope. The revised design received `Correct-to-implement`.
- Post-implementation full-diff review: `reviewer-flash-opencode-go` reported
  no Critical or Important findings and returned `Correct-to-merge`. Four Minor
  cleanup findings (worklog verdict, caller affinity diagnostic, unreachable
  diagnostic wording, and duplicate default-domain validation) were fixed and
  re-reviewed before integration.

## Verification

- `cargo check --workspace`
- `cargo check -p tenferro-cpu --no-default-features --features cpu-blas`
- `cargo check -p tenferro-cpu --features cpu-blas`
- `cargo test -p tenferro-cpu` — 805 passed
- `cargo test -p tenferro-cpu --doc` — 217 passed
- `cargo fmt --all -- --check`
- `git diff --check`

The BLAS-only unit-test binary cannot link without a concrete CBLAS provider in
this environment; the BLAS-only compile check succeeds and the default faer
suite covers the public typed rejection path.

## Remaining risks

Caller-managed admission cannot verify host scheduling or physical CPU
oversubscription across distinct pools; that responsibility is intentionally
owned by the caller and is reported in diagnostics and user documentation.
