# 2026-06-24 Validation and Safety Umbrella

## Summary

Created umbrella issue #1202 and addressed a broad batch of recent validation,
shape-arithmetic, dtype, AD, runtime, GPU, and FFI safety reports. The repairs
were handled as one batch because most reports shared the same root causes:
validation happened after graph recording or dispatch, shape-derived arithmetic
was duplicated, AD rules erased typed errors, and backend/FFI code lacked
reviewable ownership or unsafe contracts.

## Issue Ledger

- Fixed by shared validation/arithmetic hardening: #1143, #1144, #1147,
  #1152, #1153, #1157, #1160, #1161, #1162, #1163, #1165, #1166, #1167,
  #1168, #1170, #1171, #1174, #1177, #1178, #1180, #1181, #1182, #1183,
  #1184, #1187, #1188, #1189, #1190, #1193, #1194, #1195, #1196, #1198,
  and #1199.
- Fixed in linalg/AD/einsum-specific paths: #1150, #1151, #1155, #1156,
  #1191, and #1192.
- Fixed in GPU/resource/FFI paths: #1146, #1148, #1149, #1158, #1159,
  #1172, #1173, #1175, #1176, #1179, #1185, #1186, #1197, and the cuTENSOR
  pointer portion of #1091.
- Fixed in FFT: #1145.
- Fixed as source-clarity cleanup: #1154.
- Dispositioned with contract and regression evidence rather than behavior
  change: #1164. Pool acquisitions leave retention accounting before handing
  out ownership; a panic drops the in-flight buffer instead of re-pooling
  partially initialized memory, and regression coverage now fixes that
  accounting invariant.
- Still out of scope for this branch unless explicitly requested: the full
  LAPACK/cuda FFI `// SAFETY:` sweep from #1169. This branch adds targeted
  FFI safety contracts where the risk was actionable, but does not annotate
  every historical LAPACK call site.

## Decisions

- Preferred typed prepared metadata and shared validation helpers over repeated
  `validate_then_index` patterns.
- Kept eager AD graph recording behind validation so invalid public calls do
  not persist malformed ops for later reverse-mode failures.
- Preserved underlying AD/shape errors instead of converting them to `Option`
  or fixed sentinel behavior.
- Removed dtype-agnostic analytic-op shortcuts for AD seeds; constant seed
  construction now uses dtype-aware tensor helpers.
- For cache and buffer-pool findings, distinguished correctness/accounting
  bugs from intentional reuse-policy tradeoffs. Buffer-pool acquisitions remove
  retained capacity from the pool before handing ownership to a caller; panic
  paths intentionally drop in-flight buffers rather than retaining partially
  initialized memory, and tests assert the retained-byte accounting remains
  consistent.
- Added source-contract tests for high-risk patterns that are hard to exercise
  on hosted CI, especially GPU metadata, FFI pointer handling, unsafe comments,
  and release-mode fallback hazards.

## Verification

Incremental red/green verification was run for each fix. Final verification:

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test -p tenferro-tensor -p tenferro-runtime -p tenferro-ad -p tenferro-cpu -p tenferro-einsum -p tenferro-fft -p tenferro-linalg -p tenferro-gpu -p kdv_pinn`
- `cargo test -p tenferro-linalg --features autodiff`
- `cargo test -p tenferro-einsum --features autodiff`
- `cargo test -p tenferro-cpu linalg_pool_acquire_then_panic_keeps_retained_stats_consistent`
- `cargo test -p tenferro-runtime scalar_semantics`
- `cargo test -p tenferro-tensor backend_`
- `cargo test -p tenferro-xla executor`
- `CUDARC_CUDA_VERSION=12080 cargo test -p tenferro-gpu --features cuda scatter -- --ignored --nocapture`
- `cargo llvm-cov --workspace --release --json --output-path /tmp/tenferro-workspace-coverage.json`
- `python3 scripts/check-coverage.py /tmp/tenferro-workspace-coverage.json`
- `CUDARC_CUDA_VERSION=12080 cargo test --no-run --package tenferro-gpu --package tenferro-ad --package tenferro-linalg --features cuda --release`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review.json`

## Remaining Risks

- Full CUDA runtime behavior still depends on self-hosted GPU CI; local
  verification for some GPU changes is source-contract or compile-time only.
- The full #1169 historical LAPACK/cuda FFI safety-comment audit is larger than
  this branch and should be handled as a separate documentation/safety sweep.
- Issue closure should happen through the final PR so individual reports are
  closed only with the commit evidence visible to reviewers.
