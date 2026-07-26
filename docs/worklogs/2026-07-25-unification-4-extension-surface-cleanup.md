# Unification 4 extension surface cleanup

## Scope

This worklog records the cleanup pass for #1458/#1459 after the runtime and AD
execution paths had already moved onto `Runtime::run_compiled` and
`ExtensionModule` preparation. The purpose was to remove remaining compatibility
surfaces from the extension contract before proceeding to the later AD-cache and
benchmark phases.

## Context read

- Issues #1458, #1459, and #1460 current bodies on 2026-07-25.
- `crates/tenferro-internal-ops/src/ext_op.rs`
- `crates/tenferro-runtime/src/extension.rs`
- `crates/tenferro-runtime/src/extension_execution_context.rs`
- `crates/tenferro-ad/src/eager_exec.rs`
- `crates/tenferro-einsum/src/extension.rs`
- `crates/tenferro-fft/src/lib.rs`
- `crates/tenferro-linalg/src/extension.rs`
- `ext/sparse/src/extension.rs`
- `ext/tropical/src/extension.rs`
- `docs/spec/extension-op.md`
- `docs/spec/backend-contract.md`
- `docs/design/execution-engine-provider-architecture.md`

## Decisions

- `ExtensionOp` is now a pure semantic payload contract. It no longer exposes
  the `HostReference` trait or `host_reference()` execution hook.
- First-party native extension families (`tenferro-einsum`, `tenferro-fft`,
  `tenferro-linalg`) now execute only through their generated
  `ExtensionModule` / `ExtensionEngine` prepared operations.
- Sparse and tropical keep simple reference implementations, but those are
  module-local payload downcasts owned by their reference modules. The core
  extension payload no longer carries a public fallback execution hook.
- The runtime file that still housed `ExtensionExecutionContext` was renamed
  from `extension_runtime.rs` to `extension_execution_context.rs` to avoid
  preserving the retired `ExtensionRuntime` vocabulary as an internal anchor.
- AD eager missing-extension diagnostics now say "missing extension module"
  instead of "missing runtime" to match the new owner.

## Rejected alternatives

- Keeping `HostReference` as an optional compatibility hook was rejected because
  it would leave execution behavior attached to the semantic payload after
  `ExtensionModule` became the canonical execution owner.
- Keeping first-party native `HostReference` impls as test or eager shortcuts
  was rejected because they duplicated the module execution path and encouraged
  future code to bypass runtime-owned caches and preparation.

## Verification

Focused checks run after the cleanup:

```console
cargo test -p tenferro-internal-ops extension_op_contract_has_no_host_reference_execution_hook --lib -- --nocapture
cargo test -p tenferro-internal-ops extension_standard_lowering_has_no_legacy_option_shim --lib -- --nocapture
cargo test -p tenferro-runtime graph_executor_legacy_facade_is_not_public_surface --test integration -- --nocapture
cargo test -p tenferro-runtime legacy_extension_executor_registry_is_not_public_surface --test integration -- --nocapture
cargo test -p tenferro-ad tensor_read_extension_path_errors_when_runtime_family_is_missing --lib -- --nocapture
cargo test --manifest-path ext/sparse/Cargo.toml sparse_jvp_host_boundary_rejects_count_dtype_and_exact_shape --features autodiff -- --nocapture
cargo test --manifest-path ext/sparse/Cargo.toml sparse_vjp_host_boundary_rejects_count_dtype_and_exact_shape --features autodiff -- --nocapture
cargo test --manifest-path ext/tropical/Cargo.toml tropical_jvp_host_boundary_rejects_count_dtype_and_exact_shape --features autodiff -- --nocapture
cargo test --manifest-path ext/tropical/Cargo.toml tropical_vjp_host_boundary_rejects_count_dtype_and_exact_shape --features autodiff -- --nocapture
```

Focused compile and documentation checks run:

```console
cargo check -p tenferro-runtime --tests --message-format=short
cargo check -p tenferro-ad --tests --message-format=short
cargo check -p tenferro-einsum --tests --message-format=short
cargo check -p tenferro-fft --tests --message-format=short
cargo check -p tenferro-linalg --tests --message-format=short
cargo check -p tenferro-xla --tests --message-format=short
cargo check --manifest-path ext/sparse/Cargo.toml --tests --features autodiff --message-format=short
cargo check --manifest-path ext/tropical/Cargo.toml --tests --features autodiff --message-format=short
cargo check --manifest-path ext/sparse/Cargo.toml --tests --message-format=short
cargo check --manifest-path ext/tropical/Cargo.toml --tests --message-format=short
python3 scripts/check-public-error-docs.py
python3 scripts/check-doc-snippets.py --check
python3 scripts/check-guide-dependency-snippets.py
```

String audit for current production/docs surface:

```console
rg -n "HostReference|host_reference\\(|\\bExtensionRuntime\\b|\\bExtensionExecutor\\b|\\bGraphExecutor\\b|register_runtime\\b" \
  crates ext samples docs/spec docs/design README.md AGENTS.md \
  --glob '!target/**' --glob '!docs/worklogs/**' --glob '!docs/plans/**' --glob '!docs/superpowers/**'
```

The remaining hits are source-contract tests that deliberately contain retired
symbol names as forbidden strings.

## Residual risk

This pass is structural cleanup. It does not prove the #1454 terminal
performance gate. The next phase still needs #1460's AD transform-cache work
and then #1454's warm/cold, shape-churn, graph-churn, requires-grad split, and
operation-size tier benchmark attribution.
