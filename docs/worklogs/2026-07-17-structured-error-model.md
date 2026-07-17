# Structured Error Model

Date: 2026-07-17

## Scope

This work implements the approved structured error model from
`docs/superpowers/specs/2026-07-17-structured-error-model-design.md` and the
ordered plan in `docs/superpowers/plans/2026-07-17-structured-error-model.md`.
The branch starts from `origin/main` at `1ec2062e` and contains only the
approved design and plan commits before implementation.

## Classification policy

The migration applies one policy at every crate boundary:

1. Caller-controlled tensor relationships use shared typed validation payloads
   (`ShapeMismatch`, rank, axis, dtype, or `InvalidArgument`).
2. Unsupported operations and dtypes use `Unsupported`, with a typed local
   source when the domain owns a more specific reason.
3. Numerical failure (singularity, non-convergence, division by zero, and
   related numeric-domain failures) uses `NumericalFailure` and retains the
   typed local source.
4. Typed in-workspace backend/kernel failures use `BackendSource`; vendor
   status text with no typed source is the only remaining `BackendFailure` text
   case.
5. Typed file, stream, serialization, or dynamic-library failures use
   `ErrorKind::Io` and retain their source. They are not backend failures just
   because they occurred during backend execution.
6. Missing, uninitialized, poisoned, or invalid executor/cache/device state
   uses `ErrorKind::RuntimeState`, retaining a typed source when available.
   Impossible invariants remain `Internal`.

`ErrorPhase` is orthogonal to the category: eager and traced paths use the
same validation payload and classification for the same known input, while
the phase records whether the fact was discovered during graph construction,
compilation, or execution.

## Checkpoints

- `109a8d11`: shared tensor validation payloads and boxed shape mismatch.
- `13bcf73d`: tensor outer source-chain model and public error docs.
- `772d9c4e`: CPU structured validation and typed backend sources.
- `89d3ab42`: CUDA structured errors, including typed unsupported-dtype source.
- `1a9b8b4f`: runtime and default-feature extension migration.

The linalg, optional extension, audit, and release-gate verification is
recorded below; implementation commits will be added after the final staged
diff review.

## Verification notes

The CPU all-target test suite ran 311 unit and 43 integration tests before the
benchmark failed because the local environment cannot resolve the requested
NUMA/affinity placement:

```text
faer AllAllowed placement should resolve: ManagedAffinityUnavailable {
    requested: AllAllowed,
    backend: Faer,
}
```

This is an environment limitation, not a passing claim for the complete
all-target command. The isolated clean-`origin/main` WebGPU comparison is
recorded below.

## Public API documentation audit

The runtime audit initially reported 149 public `Result` APIs without a
usable concrete `# Errors` contract. Each runtime section was reviewed at the
operation boundary rather than filled with a category-only sentence. The
sections name the observable validation payloads (`ShapeMismatch`,
`RankMismatch`, `AxisOutOfBounds`, `DuplicateAxis`, `DTypeMismatch`, and
`InvalidArgument`), operation-level unsupported cases, numerical failures,
typed backend/extension sources, and runtime-state failures where those paths
are reachable. Traced operations additionally describe which checks happen at
graph build and which symbolic constraints or bound indices can fail at
compile/execution, under a separate `# Deferred errors` heading.

The repository gate is `scripts/check-public-error-docs.py`. It audits public
functions and public trait methods returning `Result`, understands both
`///` and proc-macro `#[doc = ...]` documentation, rejects a missing or
category-only `# Errors` section, and requires a `# Deferred errors` section
when a traced API documents deferred validation. Its unit tests are in
`scripts/test-check-public-error-docs.py`; CI runs the unit test, the complete
workspace audit, and the audit of lines changed from the event base. The
Clippy job also explicitly denies both
`clippy::missing_errors_doc` and `clippy::missing_panics_doc` for the workspace,
`ext/tropical`, and `ext/sparse`. This makes future public `Result` APIs fail
the source audit/CI instead of relying on a long-lived lint allowlist.

## Boundary mapping audit

The implementation was audited by source owner and failure meaning, not by
call-site compilation alone:

- Tensor-core owns the shared validation vocabulary. `ShapeMismatch` is boxed
  at the outer tensor boundary to keep error values below the large-error lint
  threshold while preserving `matches!`, `source()`, and
  `ShapeMismatch::... .into()` ergonomics.
- Operation-level unsupported dtypes use `UnsupportedDType { op, dtype }` and
  a typed CUDA/WebGPU/linalg source. `UnsupportedDTypeConversion` is reserved
  for a real input-to-target conversion or cast; no operation-only rejection
  is represented as `from == to`.
- Singularity, non-convergence, zero divisors, and negative integer powers are
  numerical failures with local typed sources. CUDA/cuTENSOR/cuSOLVER/LAPACK
  status and workspace failures retain typed provider sources and are backend
  failures. CPU placement discovery, unavailable managed affinity, and
  unknown NUMA nodes are runtime-state failures; unsupported external affinity
  is an unsupported extension condition, not a backend catch-all.
- Typed file, stream, serialization, and dynamic-library failures use the I/O
  classification. Missing/uninitialized/poisoned registry, executor, cache,
  device, and buffer state uses runtime-state. `BackendFailure(String)` is
  retained only for boundaries that genuinely expose no typed source and for
  explicit test injection; production typed boundaries use `backend_source` or
  an extension source. The AD adapter necessarily converts the typed runtime
  error to the external `tidu::ADRuleError` message-only type at that foreign
  API boundary, while retaining the typed runtime error in deferred errors.

## Optional feature and dependency evidence

These checks passed with the structured model:

```text
cargo check -p tenferro-cpu --no-default-features --features cpu-blas --all-targets --message-format=short
cargo check -p tenferro-linalg --no-default-features --features cpu-blas --all-targets
cargo clippy -p tenferro-linalg --no-default-features --features cpu-blas --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo check -p tenferro-runtime --no-default-features --features cpu-blas --all-targets --message-format=short
cargo check -p tenferro-gpu --no-default-features --features 'cuda cpu-faer' --all-targets --message-format=short
cargo check -p tenferro-xla --no-default-features --features pjrt --all-targets --message-format=short
cargo clippy --manifest-path ext/sparse/Cargo.toml --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
```

The correct CUDA feature check includes `cpu-faer`; the intentionally
incomplete command
`cargo check -p tenferro-gpu --no-default-features --features cuda --all-targets`
fails at the repository's required CPU-fallback `compile_error!`, so it is not
used as a feature-support claim.

WebGPU remains unavailable for a pre-existing dependency-resolution reason.
To prove that it was not introduced here, an isolated clean worktree at
`/private/tmp/tenferro-origin-main-webgpu-1784299674-45394` was checked at
clean `origin/main` commit `1ec2062e` with the exact command:

```text
cargo check -p tenferro-gpu --features webgpu --all-targets --message-format=short
```

Its first causal errors were `TensorBinding<_>` versus
`TensorBinding<WgpuRuntime>`, followed by `cubecl_ir::StorageType` and type
mismatches (12 errors). The current branch's
`cargo check -p tenferro-gpu --no-default-features --features 'webgpu cpu-faer' --all-targets --message-format=short`
reproduces the same CubeCL binding/storage mismatch (18 diagnostics, with the
same first error family). The clean main resolution uses direct CubeCL commit
`b9e8f0f3...`, cubek commit `7d9e382...`, and CubeCL dependency
`6424d9da...`; this PR does not alter those revisions.

The documented CUDA ignored-test command was not run because this macOS host
has no `/usr/local/cuda-12.8` toolkit or NVIDIA runtime:

```text
CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.8 LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-gpu --features cuda -- --ignored
```

The full CPU all-target command is likewise not claimed as passing because
the local Faer `AllAllowed` affinity test returns the
`ManagedAffinityUnavailable` runtime-state error shown above. The focused CPU
library/integration commands pass. The docs-site checker passes under
Python 3.11 and 3.12; the host default `python3` is 3.9.6 and exits with the
checker’s explicit “Python 3.11+ is required” diagnostic.

The first post-migration `cargo test --workspace --release` also exposed two
environment-sensitive test/documentation issues during verification. The
XLA PJRT execution test could race the `pjrt_env` tests: both ran in one test
binary, but each used a different static mutex while `pjrt_env` temporarily
set `TENFERRO_PJRT_PLUGIN=/definitely/missing/pjrt.so`. The execution test then
observed that temporary value and unwrapped a deliberate `PluginLoad` error.
The fix is a shared parent-module lock used by both modules; the full
`cargo test -p tenferro-xla --features pjrt --test integration --release`
command passes with all 36 tests.

The same release run found two einsum doctests whose trailing `Ok::<...>`
annotations still named the removed AD error type after the einsum API became
crate-local. The examples now return `tenferro_einsum::Error`; the exact
`cargo test -p tenferro-einsum --doc --release` gate passes all 81 doctests.

The next workspace release run found one remaining stale doctest in
`tenferro_tensor::validate::DiagonalError`: the example constructed the
non-batched `SingularOrNonFinite` variant with the removed `batch` field. The
example now uses the current `{ index }` payload, and
`cargo test -p tenferro-tensor --doc --release` passes all 284 doctests.

The complete release gate was then rerun exactly as
`cargo test --workspace --release`; it exited successfully. The resulting
workspace run included the XLA integration suite (36 tests), runtime doctests
(238), and tensor doctests (284), with no failed tests or doctests. Coverage
was collected with
`cargo llvm-cov --workspace --release --json --output-path coverage.json` and
passed `python3.11 scripts/check-coverage.py coverage.json` for 163/163 files
(three excluded by policy).

The remaining static gates also passed: `cargo fmt --all --check`,
`git diff --check`, `cargo doc --workspace --no-deps`, full and
`--changed-from origin/main` public-error documentation audits, the audit's
nine unit tests, docs consistency tests, repository-rule review tests, the
workspace strict Clippy command, and strict Clippy for both nested extension
manifests. The public-error audit reported `public-error-docs-ok` in both
modes; no `result_large_err` or documentation allowlist was added.

The audit was then tightened after the committed-head repository review
identified eight `TypedTensorView` methods and four CUDA cache methods whose
sections were either generic or separated from the heading by a missing blank
doc line. The concrete-variant matcher now rejects `Error::Validation` unless
the section also names a payload variant or an observable condition; its
negative regression test covers the exact category-only wording. The affected
accessors now name rank/index/layout/dtype/runtime-state conditions, the linalg
option methods name their input, gauge, provider, and placement failures, and
the eager pad/extension and CubeCL launch docs name their concrete sources.
The exact follow-up checks passed:

```text
python3.11 scripts/test-check-public-error-docs.py       # 9 tests
python3.11 scripts/check-public-error-docs.py            # public-error-docs-ok
python3.11 scripts/check-public-error-docs.py --changed-from origin/main
python3 scripts/test-doc-consistency.py
python3 scripts/test-repository-rules-review.py
cargo doc --workspace --no-deps
cargo test -p tenferro-ad --doc --release                    # 130 passed
cargo test -p tenferro-linalg --doc --release                # 49 passed
```

The worktree-inclusive repository-rules review was rerun with
`python3 scripts/repository-rules-review.py --base origin/main --head HEAD
--worktree --output-json /tmp/repository-rules-review-worktree.json` and
returned `Verdict: pass / No findings`. Rustdoc produced no broken-link
warnings after replacing cross-crate links with the public
`tenferro_tensor::ValidationError` path.

After checkpoint `05221f5a`, the committed-head gates were rerun without a
worktree overlay. `cargo check --workspace --all-targets
--message-format=short` exited 0, as did the CI-identical commands:

```text
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo clippy --manifest-path ext/sparse/Cargo.toml --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review-final.json
```

The final committed-head review again returned `Verdict: pass / No
findings`, and `git diff HEAD^ HEAD --check` was clean.

The current WebGPU feature check was rerun as
`cargo check -p tenferro-gpu --no-default-features --features 'webgpu cpu-faer' --all-targets --message-format=short` and still exits 101 with 18 diagnostics. Its first errors remain the CubeCL `TensorBinding`, `StorageType`, and `Type` mismatches recorded above; this is the same dependency mismatch proven on clean `origin/main`, not a structured-error change.

The hardware-independent CUDA feature test command
`cargo test -p tenferro-gpu --no-default-features --features 'cuda cpu-faer' --all-targets`
passes 38 tests and explicitly skips 108 GPU-dependent tests. The exact CUDA
`-- --ignored` command remains unrun because the host lacks the required
toolkit/runtime. The two optional CPU-BLAS test commands were rerun and still
fail before tests at the arm64 linker with missing BLAS/LAPACK symbols; their
`cargo check` and strict Clippy gates pass.

The optional BLAS test commands were attempted but are not claimed as
passing on this macOS arm64 host:

```text
cargo test -p tenferro-linalg --no-default-features --features cpu-blas --all-targets
cargo test -p tenferro-runtime --no-default-features --features cpu-blas --all-targets
```

Both fail during linking, before tests run. The first causal linker symbols
are `_cblas_ctrsm`, `_cblas_dtrsm`, `_cblas_strsm`, `_cblas_ztrsm`, and
LAPACK symbols such as `_cgetrf_`/`_dgetrf_`; the runtime command additionally
reports `_cblas_cgemm`/`_cblas_dgemm`/`_cblas_sgemm`/`_cblas_zgemm`. No matching
arm64 BLAS/LAPACK provider is configured in this environment. The same
features do pass `cargo check` and strict lints. CUDA-feature non-ignored
tests compile and pass, while the CUDA tests marked `ignored` remain
hardware-dependent and were not run.

The committed-head repository-rules review then identified two documentation
quality issues in the public surface: the four `GraphCompilerEinsumExt` methods
had concrete `# Errors` text but no method-specific summary before the section,
and several tensor-error doctests relied on imports or type aliases that a
reviewer could not verify from the example alone. The methods now describe
their exact default/explicit optimizer and textual/parsed-subscript behavior.
The tensor-error examples use fully qualified public paths and match concrete
variants, including the typed backend source chain, so their compilation and
runtime contract is visible without inference. The targeted doctests pass
(284 tensor examples and 78 einsum examples), and the committed-head review
was rerun after these changes.

The follow-up review also rejected the remaining blanket view/tensor wording
as too broad. The tensor public surface was audited by operation family rather
than patched call-site by call-site: strided views now name rank/slice/layout
and mutable-overlap variants; owned views name reshape, broadcast, and slice
payloads; typed/dynamic tensors distinguish dtype mismatch, host-access runtime
state, shape-data length, index, and layout failures; and `TensorRead`/
`TensorWrite` name stride and offset arithmetic failures. The full workspace
docs build and the worktree repository-rules review now return no findings.

## Residual risks

Hardware-dependent CUDA/WebGPU/PJRT execution and hosted affinity/NUMA
benchmarks require their documented environments. No dependency revision or
lint allowlist is added to hide those limitations.
