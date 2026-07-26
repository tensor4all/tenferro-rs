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
quality issues in the public surface: the four `TraceContextEinsumExt` methods
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

## Review remediation on 2026-07-18

The follow-up review was handled as boundary-level changes rather than
per-call-site compile fixes:

- Traced unary, binary, and ternary construction now uses fallible dtype
  inference at `GraphBuild`. Ordered operations, including complex `Rem`,
  return a typed `Unsupported` error with the graph-build phase instead of
  reaching an inference panic. The compiler keeps the same inference helper
  on the `Compile` phase, and the public-operation audit found no remaining
  infallible dtype-inference helper.
- `BroadcastError` has one conversion to the shared validation vocabulary.
  Eager, traced, and direct runtime tensor surfaces use that conversion for
  `IncompatibleBinary`, `IncompatibleInput`, and `RankTooLarge`; parity tests
  assert identical payloads while allowing the discovery phase to differ.
- `ShapeGuardFailure` wraps the original typed `ShapeGuardError`. Its
  autodiff-only side channel records that value before it crosses tidu's
  message-only `ADRuleError` boundary; the eager and traced frontends consume
  the side channel and expose `Error::AdRuleSource` with a real source chain.
  Non-autodiff builds compile without a dead side-channel field.
- Einsum planning now owns a typed `PlanningError`: caller-invalid
  expressions, shapes, paths, and optimizer options classify as validation,
  while an explicitly unavailable or poisoned planner state classifies as
  runtime state. Extension lowering passes the source's `kind()` through the
  typed `ExtensionLoweringError`, so XLA no longer flattens unsupported,
  numerical, backend, or runtime-state failures into invalid argument.
- Online guides and `TracedTensor::add` explain why `a + b + c` cannot compose
  when `Add<Output = Result<_>>`, show the two-step `?` form and the explicit
  method chain, and state that robust error handling takes priority over
  operator-chain concision. Add/sub docs no longer claim zero-divisor errors;
  div/rem retain their execution-time numerical condition.

The focused review-fix verification completed before the final workspace gate:

```text
cargo test -p tenferro-runtime --all-targets --no-fail-fast --message-format=short
cargo test -p tenferro-ad eager_and_traced_broadcast_errors_share_payloads_across_discovery_phases --test integration --message-format=short
cargo test -p tenferro-internal-ops --all-targets --no-fail-fast --message-format=short
cargo test -p tenferro-ad ad_rule_error --lib --message-format=short
cargo test -p tenferro-einsum --test integration error_public --message-format=short
cargo test -p tenferro-xla error::tests --lib --message-format=short
cargo test -p tenferro-internal-ops --doc --message-format=short
cargo test -p tenferro-einsum -p tenferro-xla --doc --message-format=short
```

All commands above exited successfully. The implementation adds no
compatibility variants, deprecated aliases, or long-lived lint allowlists;
staged file lists and `git diff --cached --check` were reviewed before each
checkpoint commit.

## Final review-fix gates after the latest-main rebase

The branch was rebased from merge-base `1ec2062e` onto the current
`origin/main` `83f7c48b` after local preflight rejected the stale base. The
rebase had no conflicts; `git merge-base HEAD origin/main` now returns
`83f7c48b`.

The complete release and coverage gates passed:

```text
cargo test --workspace --release
# exit 0; all workspace unit, integration, and doctests passed
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3.11 scripts/check-coverage.py coverage.json
# exit 0; Coverage check: 163/163 files passed (excluded: 3)
```

The optional compile gates also passed:

```text
cargo check -p tenferro-cpu --no-default-features --features cpu-blas --all-targets --message-format=short
cargo check -p tenferro-linalg --no-default-features --features cpu-blas --all-targets
cargo check -p tenferro-runtime --no-default-features --features cpu-blas --all-targets --message-format=short
cargo check -p tenferro-gpu --no-default-features --features 'cuda cpu-faer' --all-targets --message-format=short
cargo check -p tenferro-xla --no-default-features --features pjrt --all-targets --message-format=short
```

The CUDA feature's hardware-independent tests passed with 38 passed and 108
ignored. The following BLAS test commands were run but are not claimed as
passing: both stop at the arm64 linker because this host lacks matching BLAS
and LAPACK symbols. The first causal symbols are `_cblas_ctrsm`,
`_cblas_dtrsm`, `_cblas_strsm`, `_cblas_ztrsm`, `_cgetrf_`, and `_dgetrf_` for
linalg; the runtime test additionally reports `_cblas_cgemm`,
`_cblas_dgemm`, `_cblas_sgemm`, and `_cblas_zgemm`.

```text
cargo test -p tenferro-linalg --no-default-features --features cpu-blas --all-targets
cargo test -p tenferro-runtime --no-default-features --features cpu-blas --all-targets
```

The current CI profiles were exercised with Python 3.11 on this macOS host:

```text
PATH=/Users/hiroshi/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin:$PATH bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-runtime remainder_rejects_complex_dtype_at_graph_build_without_panicking' --ci-profile docs
# exit 0; docs-site and all 9 stages passed (optional Graphviz dependency graph skipped: dot not installed)
PATH=/Users/hiroshi/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin:$PATH bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-runtime remainder_rejects_complex_dtype_at_graph_build_without_panicking' --ci-profile workspace-faer
# exit 0; nextest 2243/2243 passed (one leaky test reported), workspace release doctests passed
```

The CI unit tests initially exposed a macOS Bash 3.2 bug in
`check-pr-fast.sh`: `set -u` rejects the length expansion of an empty array.
The script now tracks explicit focused-test/profile counts and only expands
non-empty arrays. This keeps the existing CI policy intact while making the
preflight portable. With the fix, the focused CI policy tests pass:

```text
PATH=/Users/hiroshi/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin:$PATH /Users/hiroshi/.local/bin/python3.11 -m unittest discover -s scripts/ci/tests -p 'test_change_policy.py' -v
# exit 0; 16 tests passed
```

The full `ci-config` profile reaches its final `actionlint` step after 81 CI
unit tests pass, but this host has no `actionlint` executable (`actionlint:
command not found`). It is therefore an environmental limitation and is not
claimed as a passing local gate. The obsolete `--ci-profile local-gate`
argument was also verified to be rejected because current `origin/main`
removed that profile; the current profiles above are authoritative.

## Independent re-review follow-up on 2026-07-18

The follow-up review identified one semantic overvalidation and two small
portability/documentation gaps. The structural-equality check in
`validate_broadcast_in_dim_args` was removed: when the target extent is
symbolic, a concrete input extent is now left for the existing deferred shape
guard, while known-known incompatible extents still fail during graph build.
The regression constructs a concrete `[2]` input and a symbolic extent owned by
another tensor, verifies execution for the compatible `[2]` binding, and
verifies an incompatible `[3]` binding returns the typed shape mismatch at
execution. A separate test asserts the known-known `[2]` to `[3]` mismatch is
reported with `ErrorPhase::GraphBuild`.

The traced `div` and `rem` deferred-error sections now state that integer zero
divisors are execution-time `Error::TensorRuntime` numerical failures whose
typed backend source remains in the source chain; floating-point and complex
zero divisors retain their numeric semantics.

The fast preflight now uses explicit changed/untracked-file counters before
expanding arrays, covering both changed-file loops under macOS Bash 3.2's
`set -u` behavior. The new no-change regression invokes `/bin/bash` directly
against a clean temporary repository with `--base HEAD`; because this host's
system Python 3.9 cannot import the repository's `StrEnum`-based policy
module, the test prepends the resolved Python 3.11 runtime directory so the
script still executes the same `python3` entry point.

The focused RED/GREEN and final gates were:

```text
cargo test -p tenferro-runtime broadcast_in_dim_sym --lib
# exit 0; 2 tests passed
PATH=/Users/hiroshi/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin:$PATH \
  /Users/hiroshi/.local/bin/python3.11 -m unittest discover -s scripts/ci/tests -p 'test_change_policy.py' -v
# exit 0; 16 tests passed
cargo test -p tenferro-runtime --all-targets --no-fail-fast --message-format=short
# exit 0; 247 unit tests and 41 integration tests passed
cargo test --workspace --release
# exit 0; all workspace unit, integration, and doctests passed
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
# exit 0
cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo clippy --manifest-path ext/sparse/Cargo.toml --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
# both exit 0
cargo fmt --all --check
/bin/bash -n scripts/check-pr-fast.sh
# both exit 0
PATH=/Users/hiroshi/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin:$PATH \
  /bin/bash scripts/check-pr-fast.sh --base origin/main --no-fetch --coverage-reviewed \
  --test 'cargo test -p tenferro-runtime broadcast_in_dim_sym_defers_cross_tensor_symbolic_extent_validation'
# exit 0; fast PR checks passed
```

## CI changed-base audit fix on 2026-07-18

The public-error documentation audit keeps its `revision...HEAD` comparison:
the triple-dot form is required because the audit must inspect the merge-base
diff and must fail loudly when the requested base object is unavailable. PR
CI's `clippy` job was invoking `--changed-from` after the default
`actions/checkout` shallow checkout, so a base SHA such as
`e8ef92c9e62260eb83e2c1e221234d945e9e9c3c` was not guaranteed to be present.
The correct boundary fix is `fetch-depth: 0` on that checkout. A workflow
contract test scans every workflow job that invokes `--changed-from` and
requires this full-history setting; no two-dot fallback or audit error
suppression was added.

The public-error audit tests also create a two-commit repository, clone only
the head with `git clone --depth 1`, and verify that the missing-base
`git diff base...HEAD` returns exit 128 and that the Python audit propagates
the failure. The same fixture passes against the full-history repository.
This reproduces the hosted checkout failure without depending on GitHub's
runner filesystem.

The failing PR run `29627808652` confirmed the same sequence: the full audit
printed `public-error-docs-ok`, then the changed audit raised
`CalledProcessError` for
`git diff --name-only e8ef92c9e62260eb83e2c1e221234d945e9e9c3c...HEAD -- *.rs`
with `returned non-zero exit status 128`. The checkout-depth change addresses
that Git object availability failure at the workflow boundary. The same run's
configuration checks passed; a separate extension test lane failed on
tropical tests asserting the old validation payload and is not silenced or
folded into this checkout fix.

## Tropical AD host-boundary classification follow-up on 2026-07-18

The two failing tropical tests were stale expectations, not an implementation
classification defect. The structured implementation already validates the
three cases in this order and preserves the shared error vocabulary:

| Surface | Input defect | Required result |
| --- | --- | --- |
| JVP and VJP host reference | wrong input arity/configuration | `Validation(InvalidArgument { argument: "configuration" })` |
| JVP tangent or VJP cotangent | dtype differs from the primal/active input | `Validation(DTypeMismatch { expected: F64, actual: I64 })` |
| JVP tangent or VJP cotangent | exact shape differs from the required shape | `Validation(ShapeMismatch(IncompatibleShapes { lhs: expected, rhs: actual }))` |

The tests now assert each operation name, `ErrorKind`, concrete payload, and
the `thiserror` source chain while retaining the `catch_unwind` boundary
regression. Operation-level unsupported dtype remains a distinct typed
tropical extension error when the primal inputs themselves use an unsupported
but matching dtype; it is not substituted for a tangent/cotangent dtype
mismatch. The direct `HostReference` contract returns the tensor error, while
the runtime execution wrapper preserves that typed source in
`Error::TensorRuntime` and supplies the execution phase.

The focused RED/GREEN and strict feature-enabled gates were:

```text
cargo test --manifest-path ext/tropical/Cargo.toml --features autodiff --lib tropical_jvp_host_boundary_rejects_count_dtype_and_exact_shape -- --nocapture
# initial RED: stale InvalidArgument assertion failed
# final GREEN: 1 passed
cargo test --manifest-path ext/tropical/Cargo.toml --features autodiff --lib tropical_vjp_host_boundary_rejects_count_dtype_and_exact_shape -- --nocapture
# exit 0; 1 passed
cargo test --manifest-path ext/tropical/Cargo.toml --features autodiff --all-targets -- --nocapture
# exit 0; all tropical unit, integration, and benchmark targets passed
cargo clippy --manifest-path ext/tropical/Cargo.toml --features autodiff --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
# exit 0
```

The feature-enabled Clippy gate also exposed two pre-existing lint findings
inside the same tropical validation/AD module: an unused `enumerate()` index
and an elidable helper lifetime. Both were removed without changing behavior
or error mapping.

## Sparse AD host-boundary follow-up on 2026-07-18

The first rerun of the hosted `extensions` profile confirmed that the tropical
tests were fixed, then exposed the analogous stale sparse JVP/VJP assertions.
Both sparse tests still expected `InvalidArgument` for count, dtype, and shape.
The host path already used the shared value-tensor validator for tangent and
cotangent inputs, so the observed dtype and shape results were the intended
structured errors rather than an implementation regression.

The owning sparse metadata boundary was then audited as well. Its primal,
tangent, and cotangent metadata checks had been erasing dtype and rank facts
into the configuration bucket. They now use the same exact constructors as
the host boundary: invalid arity/active-input configuration remains
`InvalidArgument`, wrong dtype is `DTypeMismatch`, wrong rank is
`RankMismatch`, and known value extent is `ShapeMismatch`. The tests assert
the concrete payloads, coarse `ErrorKind`, operation name, and source chain
for the host path, plus shared metadata dtype/rank classification.

The sparse RED/GREEN gates were:

```text
cargo test --manifest-path ext/sparse/Cargo.toml --features autodiff --lib sparse_jvp_host_boundary_rejects_count_dtype_and_exact_shape -- --nocapture
# initial RED: stale InvalidArgument assertion failed with Validation(DTypeMismatch)
cargo test --manifest-path ext/sparse/Cargo.toml --features autodiff --lib extension::tests::sparse_ -- --nocapture
# exit 0; 3 focused classification tests passed
cargo test --manifest-path ext/sparse/Cargo.toml --features autodiff --all-targets -- --nocapture
# exit 0; 3 unit, 6 AD integration, and 2 constructor tests passed
cargo clippy --manifest-path ext/sparse/Cargo.toml --features autodiff --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
# exit 0
```

## CUDA validation-flag source follow-up on 2026-07-18

The rerun of the hosted repository-rules review identified three CUDA
validation-flag branches that had been converted to `Error::Internal(String)`
during the structured migration. The same boundary also contained an integer
domain flag branch with the same erasure, although the model did not report it.
These are backend invariant failures: the host download returned a dtype other
than the dtype used to allocate the typed validation flag. They are not caller
configuration errors, so the canonical mapping is `BackendFailure` with a
typed `CudaError::UnexpectedValidationFlagDType` source. The operation name,
expected dtype, and actual dtype remain machine-readable through the source
chain; no message parsing or compatibility variant was added.

`CudaExtensionCache::new` now documents its default 16-entry bound, and
`clear` states its poisoned-lock runtime-state condition. The six
`EagerTensor` methods named by the same review already contain concrete
`# Errors` sections in the branch source; the deterministic public-error audit
passes, so no duplicate or boilerplate documentation was added for the
review bot's contradictory LLM output.

Focused verification:

```text
cargo test -p tenferro-gpu --features cuda --lib cubecl::error::tests -- --nocapture
# exit 0; 4 typed CUDA error/source tests passed
cargo check -p tenferro-gpu --features cuda --lib --message-format=short
# exit 0
cargo clippy -p tenferro-gpu --features cuda --lib -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
# not a passing gate: the current toolchain reports 22 unscoped CUDA-module
# lint findings across dispatch/FFI/fusion/kernel code; none are silenced by
# this change
```

The hosted rerun of the review bot still reported a changing set of false
missing-`# Errors` findings despite the sections being present. This remains a
review-service limitation rather than a source omission; the local
deterministic review and public-error audit are the authoritative local gates.

## Residual risks

Hardware-dependent CUDA/WebGPU/PJRT execution and hosted affinity/NUMA
benchmarks require their documented environments. No dependency revision or
lint allowlist is added to hide those limitations.
