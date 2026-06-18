# Terasaki Bug Batch 2

## Summary

Fixed the remaining substantive `terasakisatoshi` bug reports in one
non-squash PR batch after the previous batch merged. The final scope is a
same-root-cause remediation sweep across runtime shape validation, public
fallible APIs, AD metadata and linalg rules, eager poison handling, GPU
residency/FFI checks, CPU buffer-pool initialization contracts, and repository
workflow guidance.

The durable rule learned from this batch is general: when a potentially
dangerous-looking operation is intentional, add a nearby comment, rustdoc note,
or source-contract test explaining the invariant instead of only dismissing the
report as a false positive. New audit rules should be merged into existing
rules where possible so the rule set stays usable.

## Context Read

- `AGENTS.md`, `CONTRIBUTING.md`, `REPOSITORY_RULES.md`
- `ai/contribution-workflows/bugfix-pr.md`
- `ai/contribution-workflows/repository-remediation.md`
- Previous work log `docs/worklogs/2026-06-17-terasaki-bug-batch.md`
- Current open `terasakisatoshi` issue list from GitHub
- Mini-agent audits for public panic/API, GPU/FFI residency, and workflow rules

## Classification Ledger

- Fixed in this PR: #1076, #1082, #1084, #1088, #1089, #1090, #1091, #1092,
  #1093, #1094, #1095, #1096, #1097, #1098, #1099, #1100, #1101, #1102,
  #1103, #1104, #1105, #1106, #1107, #1108, and #1109.
- False positive with source-contract or source comments: the #1107
  `dot_conj_folding` concern. The pass moves `Conj` through transparent layout
  ops, while the `DotGeneral` operand remains wired to the layout output.
- Out of this bug-fix PR: #1054 is documentation/enhancement work, not a
  substantive bug in this batch.

## Fixed Patterns

- Runtime and tensor shape arithmetic now uses fallible checked paths for
  `DimExpr` evaluation, shape inference, graph compilation, accessors, tensor
  constructors, and fused einsum dimension products.
- Public panic surfaces now have fallible alternatives and compatibility-wrapper
  comments where the panic API remains: tensor accessors/constructors,
  `extension::apply`, traced symbolic axis helpers, reductions, tropical
  composition helpers, CPU context/cache helpers, and CUDA extension cache
  helpers.
- AD structural/indexing/linalg rules fail closed on malformed metadata,
  symbolic non-concrete shapes, invalid rank, mismatched scatter/gather window
  metadata, non-square LU variants, and invalid extension metadata instead of
  panicking.
- Eager AD/runtime paths return typed errors for backend or extension poison,
  backend execution errors, invalid recorded graph inputs, and shape-packing
  upload failures.
- GPU CUDA/CubeCL/WebGPU paths now validate runtime/device residency before
  zero-sized fast paths, downloads, raw device pointer exposure, GEMM/cuTENSOR
  FFI, scatter launches, and linalg downloads. FFI/provider assumptions carry
  local `SAFETY` comments or source-contract tests.
- CPU buffer-pool acquisition separates uninitialized/stale output buffers from
  zeroed buffers. Every uninitialized-pool acquisition has a one-line
  write-before-read rationale; read-before-write faer paths use zeroed storage.
- Borrowed graph-executor slot workspace no longer retypes a
  `Vec<Option<ExecSlot<'static>>>` allocation with `Vec::from_raw_parts`.
  Borrowed-input execution uses a lifetime-local workspace and retains capacity
  separately.
- Runtime shape validation now rejects duplicate gather dims, invalid
  concatenate non-axis dimensions, unreadable backend default tensor equality,
  and malformed terminal lazy-view instruction arity before execution while
  still allowing symbolic shape-reference inputs for dynamic reshape/broadcast
  arity.

## Workflow And Rule Updates

- `repository-remediation.md` is mandatory reading from the bug-fix workflow.
- `tenferro-bugfix-pr` adapters cover Codex, Claude Code, OpenCode, and Kimi
  CLI, and route related issue batches to the remediation workflow.
- Bug-fix workflow guidance now requires same-root-cause/same-pattern searches,
  one non-squash PR for related batches, audit-rule proposals when useful, and
  source comments or source-contract tests for false positives whose invariants
  are not obvious.
- Before adding a new audit/repository rule, agents must inventory nearby rules
  and merge, tighten, or relocate overlapping guidance where possible.

## Verification

Final local verification before push:

- `cargo check -p tenferro-tensor -p tenferro-internal-ops -p tenferro-runtime -p tenferro-einsum -p tenferro-fft -p tenferro-linalg --features tenferro-linalg/autodiff`
- `cargo test -p tenferro-runtime --lib`
- `cargo test -p tenferro-runtime --test extension_runtime`
- `cargo test -p tenferro-tensor --lib`
- `cargo test -p tenferro-internal-ops --lib`
- `cargo test -p tenferro-cpu --lib`
- `cargo test -p tenferro-cpu --features provider-inject --test inject_tests`
- `cargo test -p tenferro-ad --lib`
- `cargo test -p tenferro-einsum --lib`
- `cargo test -p tenferro-einsum --features autodiff --lib`
- `cargo test -p tenferro-fft --lib`
- `cargo test -p tenferro-linalg --features autodiff --lib`
- `cargo test -p tenferro-gpu --test cubecl_launch_contract`
- `cargo test -p tenferro-gpu --features webgpu --test webgpu_backend_contract`
- `cargo test -p tenferro-linalg --test gpu_linalg_source_contract`
- `cargo check -p tenferro-gpu --features webgpu --lib`
- `cargo check -p tenferro-gpu --features cuda --lib`
- `cargo check -p tenferro-cpu --features provider-inject --lib`
- `cargo check -p tenferro-linalg --features cuda,provider-inject,autodiff --lib`
- `cargo test` in `ext/tropical`
- `cargo test --features autodiff` in `ext/tropical`
- `cargo fmt --all --check`
- `cargo fmt --all --check` in `ext/tropical`
- `git diff --check`

No cloud GPU run was used for debugging this batch; GPU coverage here is local
source-contract tests plus feature compilation.
